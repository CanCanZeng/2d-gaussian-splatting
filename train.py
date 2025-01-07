#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lib_bilagrid import BilateralGrid, slice

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def save_ply(positions, colors, normals, file_path):
    num_points = positions.shape[0]

    with open(file_path, 'w') as f:
        # 写入PLY文件头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")

        # 写入点数据
        for i in range(num_points):
            x, y, z = positions[i].tolist()
            r, g, b = colors[i].tolist()
            nx, ny, nz = normals[i].tolist()
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)} {nx} {ny} {nz}\n")


def get_point_cloud(gaussians: GaussianModel, radii = None, visibility_filter = None, viewspace_point_tensor = None):
    positions = gaussians.get_xyz.detach()
    if radii != None:
        radii_2d = radii.detach().clamp_max(255).reshape(-1, 1).int()
    else:
        radii_2d = gaussians.max_radii2D.detach().clamp_max(255).reshape(-1, 1).int()
    opacities = gaussians.get_opacity.detach().reshape(-1, 1) * 100
    colors = torch.zeros_like(positions).int()
    colors[:, 0:1] = radii_2d
    colors[:, 1:2] = opacities.int()
    
    if viewspace_point_tensor != None:
        all_grads = viewspace_point_tensor.grad.detach()
        all_grads[~visibility_filter, :] = 0
        grads = torch.norm(all_grads[:, 0:2], dim=-1, keepdim=True) * 1000.0
        abs_grads = torch.norm(all_grads[:, 2:4], dim=-1, keepdim=True) * 1000.0
    else:
        grads = gaussians.xyz_gradient_accum / gaussians.denom * 1000.0
        grads[grads.isnan()] = 0.0

        abs_grads = gaussians.xyz_gradient_abs_accum / gaussians.denom * 1000.0
        abs_grads[abs_grads.isnan()] = 0.0

    grads_res = torch.sqrt(1 - grads*grads - abs_grads*abs_grads)
    normals = torch.cat([grads_res, grads, abs_grads], dim=-1)
    
    return positions, colors, normals


def mvs_vis_trim(gaussians: GaussianModel, scene: Scene, pipe, background):
    observe_the = 2
    observe_cnt = torch.zeros_like(gaussians.get_opacity)
    for view in scene.getTrainCameras():
        render_pkg_tmp = render(view, gaussians, pipe, background)
        out_observe = render_pkg_tmp["visibility_filter"]
        observe_cnt[out_observe] += 1
    prune_mask = (observe_cnt < observe_the).squeeze()
    if prune_mask.sum() > 0:
        gaussians.prune_points(prune_mask)


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, load_iteration = -100):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    if load_iteration and load_iteration >= -1:
        first_iter = load_iteration
    else:
        load_iteration = None

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    gaussians.training_setup(opt)
    if opt.exposure_compensation:
        exp_grids = BilateralGrid(len(scene.getTrainCameras())).cuda()
        exp_grids.train()
        exp_optimizer = torch.optim.Adam(exp_grids.parameters(), lr=0.001, betas=[0.9, 0.99], eps=1e-15)
        exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(exp_optimizer, gamma=0.01**(1.0/opt.iterations))
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    should_save_ply = False
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if opt.exposure_compensation:
            exp_scheduler.step()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        ssim_loss = (1.0 - ssim(image, gt_image))
        if opt.exposure_compensation and ssim_loss < 0.4:
            width, height = image.shape[2], image.shape[1]
            grid_x, grid_y = torch.meshgrid(torch.arange(width, device="cuda").float(), torch.arange(height, device="cuda").float(), indexing="xy")
            pix_xy = torch.stack([grid_x, grid_y], dim=-1)
            pix_xy = pix_xy + 0.5
            pix_xy[..., 0] /= width
            pix_xy[..., 1] /= height
            image = image.permute([1, 2, 0])
            image = slice(exp_grids, pix_xy, image, torch.tensor([viewpoint_cam.uid]).cuda())["rgb"]
            image = image.permute([2, 0, 1])

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        # normal_loss = lambda_normal * (viewpoint_cam.image_weight.cuda() * (rend_normal - surf_normal).abs().sum(0)).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        exposure_compensation_tv_loss = 0
        if opt.exposure_compensation:
            exposure_compensation_tv_loss = opt.lambda_exposure_compensation_tv * exp_grids.tv_loss()

        # loss
        total_loss = loss + dist_loss + normal_loss + exposure_compensation_tv_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # if viewpoint_cam.image_name == "38397.434908":
            #     positions, colors, normals = get_point_cloud(gaussians, radii, visibility_filter, viewspace_point_tensor)
            #     save_ply(positions, colors, normals, viewpoint_cam.image_name + ".ply")
            #     should_save_ply = True

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    size_threshold = None
                    # if should_save_ply:
                    #     positions, colors, normals = get_point_cloud(gaussians)
                    #     save_ply(positions, colors, normals, viewpoint_cam.image_name + ".ply")
                    #     print("...")
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    # if should_save_ply:
                    #     positions = gaussians.get_xyz.detach()
                    #     radii_2d = gaussians.max_radii2D.detach().clamp_max(255).reshape(-1, 1).int()
                    #     colors = torch.zeros_like(positions).int()
                    #     colors[:, 0:1] = radii_2d
                    #     normals = torch.zeros_like(positions)
                    #     save_ply(positions, colors, normals, viewpoint_cam.image_name + "_densify_and_prune.ply")
                    #     print("...")
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % 2000 == 0:
            #     mvs_vis_trim(gaussians, scene, pipe, background)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                if opt.exposure_compensation:
                    exp_optimizer.step()
                    exp_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000, 60_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_iteration", type=int, default = -100)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.load_iteration)

    # All done
    print("\nTraining complete.")