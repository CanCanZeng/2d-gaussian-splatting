import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.image_utils import psnr
from argparse import ArgumentParser
from torchvision.utils import save_image
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from torch.utils.cpp_extension import load
import time
import struct
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

stereo_folder = ""

def depth2rgb(depth, mask):
    sort_d = torch.sort(depth[mask.to(torch.bool)])[0]
    min_d = sort_d[len(sort_d) // 100 * 5]
    max_d = sort_d[len(sort_d) // 100 * 95]
    # min_d = 2.8
    # max_d = 4.6
    # print(min_d, max_d)
    depth = (depth - min_d) / (max_d - min_d) * 0.9 + 0.1
    
    viridis = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 256)))
    depth_draw = viridis(depth.detach().cpu().numpy()[0])[..., :3]
    # print(viridis(depth.detach().cpu().numpy()).shape, depth_draw.shape, mask.shape)
    depth_draw = torch.from_numpy(depth_draw).to(depth.dtype).to(depth.device).permute([2, 0, 1]) * mask

    return depth_draw

def normal2rgb(normal, mask):
    normal_draw = torch.cat([normal[:1], -normal[1:2], -normal[2:]])
    normal_draw = (normal_draw * 0.5 + 0.5) * mask
    return normal_draw

def read_array(path: str):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_array(array: np.ndarray, path: str):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(
            endian_character + format_char_sequence, *data_list
        )
        fid.write(byte_data)


def render_set(model_path, use_mask, name, iteration, views, gaussians: GaussianModel, pipeline, background, write_image, poisson_depth):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    info_path = os.path.join(model_path, name, "ours_{}".format(iteration), "info")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(info_path, exist_ok=True)

    resampled = []
    psnr_all = []

    # gaussians._features_dc = gaussians._features_dc[0:1,:]
    # gaussians._features_rest = gaussians._features_rest[0:1,:]
    # gaussians._opacity = gaussians._opacity[0:1,:]
    # gaussians._rotation = gaussians._rotation[0:1,:]
    # gaussians._scaling = gaussians._scaling[0:1,:]
    # gaussians._xyz = gaussians._xyz[0:1,:]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        background = torch.zeros((3), dtype=torch.float32, device="cuda")
        render_pkg = render(view, gaussians, pipeline, background)


        image, normal, depth, opac, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["rend_normal"], render_pkg["surf_depth"], render_pkg["rend_alpha"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # save_image(image.cpu(), os.path.join(render_path, '{}'.format(view.image_name) + ".png"))

        # mask_gt = view.get_gtMask(use_mask)
        # gt_image = view.get_gtImage(background, use_mask).cuda()
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        mask_gt = torch.ones_like(opac)
        psnr_all.append(psnr(gt_image, image).mean().cpu().numpy())
        mask_vis = (opac.detach() > 1e-5)

        normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis
        d2n = render_pkg["surf_normal"]
        d2n = torch.nn.functional.normalize(d2n, dim=0) * mask_vis

        # if name == 'train':
        #     pts = resample_points(view, depth, normal, image, mask_vis * mask_gt * mask_clip)
        #     grid_mask = grid_prune(occ_grid, grid_shift, grid_scale, grid_dim, pts[..., :3], thrsh=1)
        #     clean_mask = grid_mask #* mask_mask
        #     pts = pts[clean_mask]
        #     resampled.append(pts.cpu())

        if write_image:
            normal_wrt = normal2rgb(normal, mask_vis)
            depth_wrt = depth2rgb(depth, mask_vis)
            d2n_wrt = normal2rgb(d2n, mask_vis)
            normal_wrt += background[:, None, None] * (~mask_vis).expand_as(image) * mask_gt
            depth_wrt += background [:, None, None]* (~mask_vis).expand_as(image) * mask_gt
            d2n_wrt += background[:, None, None] * (~mask_vis).expand_as(image) * mask_gt
            outofmask = mask_vis * (1 - mask_gt)
            mask_vis_wrt = outofmask * (opac - 1) + mask_vis
            img_wrt = torch.cat([gt_image, image, normal_wrt, depth_wrt], 2)
            wrt_mask = torch.cat([opac * mask_gt, mask_vis_wrt, mask_vis_wrt, mask_vis_wrt], 2)
            img_wrt = torch.cat([img_wrt, wrt_mask], 0)
            info_file_path = os.path.join(info_path, '{}'.format(view.image_name) + f".png")
            render_file_path = os.path.join(render_path, '{}'.format(view.image_name) + ".png")
            gt_file_path = os.path.join(gts_path, '{}'.format(view.image_name) + ".png")
            os.makedirs(os.path.dirname(info_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(render_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(gt_file_path), exist_ok=True)
            save_image(img_wrt.cpu(), info_file_path)
            save_image(image.cpu(), render_file_path)
            save_image((torch.cat([gt_image, mask_gt], 0)).cpu(), gt_file_path)
            
        if stereo_folder != "":
            depth_path = os.path.join(stereo_folder, "depth_maps", '{}'.format(view.image_name) + f".jpg.geometric.bin")
            os.makedirs(os.path.dirname(depth_path), exist_ok=True)
            depth_np = depth.detach().cpu().permute(1, 2, 0).numpy()
            write_array(depth_np, depth_path)
            
            normal_path = os.path.join(stereo_folder, "normal_maps", '{}'.format(view.image_name) + f".jpg.geometric.bin")
            os.makedirs(os.path.dirname(normal_path), exist_ok=True)
            normal_np = normal.detach().cpu().permute(1, 2, 0).numpy()
            write_array(normal_np, normal_path)

    # os.system(f"rm {model_path}/eval_result.txt")
    with open(f"{model_path}/eval_result.txt", 'a') as f:
        f.write(f'PSNR_{name}: {np.mean(psnr_all)}\n')

    # if name == 'train':
    #     resampled = torch.cat(resampled, 0)
    #     mesh_path = f'{model_path}/poisson_mesh_{poisson_depth}'
        
        # poisson_mesh(mesh_path, resampled[:, :3], resampled[:, 3:6], resampled[:, 6:], poisson_depth, 1 * 1e-4)



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, write_image: bool, poisson_depth: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        scales = [1]
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=scales)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_test:
             render_set(dataset.model_path, True, "test", scene.loaded_iter, scene.getTestCameras(scales[0]), gaussians, pipeline, background, write_image, poisson_depth)

        if not skip_train:
             render_set(dataset.model_path, False, "train", scene.loaded_iter, scene.getTrainCameras(scales[0]), gaussians, pipeline, background, write_image, poisson_depth)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--img", action="store_true")
    parser.add_argument("--depth", default=10, type=int)
    parser.add_argument("--stereo_folder", default='', type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    stereo_folder = args.stereo_folder
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.img, args.depth)
