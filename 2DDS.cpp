#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>


#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


Eigen::Matrix3d BuildRotation(
    const Eigen::Vector4d &r
    ) {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector4d q = r.normalized();

    double w = q[0];
    double x = q[1];
    double y = q[2];
    double z = q[3];

    R(0, 0) = 1 - 2 * (y*y + z*z);
    R(0, 1) = 2 * (x*y - w*z);
    R(0, 2) = 2 * (x*z + w*y);

    R(1, 0) = 2 * (x*y + w*z);
    R(1, 1) = 1 - 2 * (x*x + z*z);
    R(1, 2) = 2 * (y*z - w*x);

    R(2, 0) = 2 * (x*z - w*y);
    R(2, 1) = 2 * (y*z + w*x);
    R(2, 2) = 1 - 2 * (x*x + y*y);

    return R;
}

Eigen::Matrix3d BuildScalingRotation(
    const Eigen::Vector3d &s,
    const Eigen::Vector4d &r
    ) {
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    S(0, 0) = s[0];
    S(1, 1) = s[1];
    S(2, 2) = s[2];

    Eigen::Matrix3d R = BuildRotation(r);

    Eigen::Matrix3d scaled_R = R * S;
    return scaled_R;
}


void GetInputs(
    std::vector<Eigen::Vector3d> &pks,
    std::vector<Eigen::Vector3d> &scales,
    std::vector<Eigen::Vector4d> &quats,
    const size_t num_points = 8
    ) {
    size_t N = num_points * num_points;
    pks.resize(N);
    scales.resize(N);
    quats.resize(N);

    double length = 0.5;
    double step = 2 * length / (num_points - 1);
    double scale = length / (num_points - 1);
    for (int row = 0; row < num_points; row++) {
        double y = -length + row * step;
        for (int col = 0; col < num_points; col++) {
            double x = -length + col * step;

            size_t idx = row * num_points + col;
            pks[idx] = Eigen::Vector3d(x, y, 0);
            scales[idx] = Eigen::Vector3d(scale, scale, scale);
            quats[idx] = Eigen::Vector4d(1, 0, 0, 0);
        }
    }
}

void GetCameras(
    Eigen::Matrix3d &intrins,
    Eigen::Matrix4d &viewmat,
    Eigen::Matrix4d &projmat,
    size_t &width,
    size_t &height
    ) {
    width = 512;
    height = 512;

    intrins <<
        711.1111, 0.0000, 256.0000,
        0.0000, 711.1111, 256.0000,
        0.0000, 0.0000, 1.0000;

    Eigen::Matrix4d c2w;
    c2w <<
        -8.6086e-01,  3.7950e-01, -3.3896e-01,  6.7791e-01,
        5.0884e-01,  6.4205e-01, -5.7346e-01,  1.1469e+00,
        1.0934e-08, -6.6614e-01, -7.4583e-01,  1.4917e+00,
        0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00;

    viewmat = c2w.inverse();
}

// https://github.com/hbb1/diff-surfel-rasterization/issues/8
void ComputeAABB(
    Eigen::Vector2d &center_point,
    Eigen::Vector2d &extent,
    const Eigen::Matrix3d &T
    ) {
    Eigen::Vector3d T0 = T.row(0);
    Eigen::Vector3d T1 = T.row(1);
    Eigen::Vector3d T2 = T.row(2);

    Eigen::Vector3d temp(1.0, 1.0, -1.0); // not a point

    // a*x^2 + b*x + c = 0
    double a = T2.cwiseProduct(T2).dot(temp);
    double b = -2 * T0.cwiseProduct(T2).dot(temp);
    double c = T0.cwiseProduct(T0).dot(temp);

    double x_mean = -b / (2*a);  // (x1+x2)/2 = -b/(2*a)
    double x_half_extent = std::abs(std::sqrt(b*b - 4*a*c)/(2*a)); // (x1-x2)/2

    // a*y^2 + b*y + c = 0
    b = -2 * T1.cwiseProduct(T2).dot(temp);
    c = T1.cwiseProduct(T1).dot(temp);
    double y_mean = -b / (2*a);
    double y_half_extent = std::abs(std::sqrt(b*b - 4*a*c)/(2*a));

    center_point = Eigen::Vector2d(x_mean, y_mean);
    extent = Eigen::Vector2d(x_half_extent, y_half_extent);
}

void Setup(
    std::vector<Eigen::Matrix3d> &Ts,
    std::vector<Eigen::Vector3d> &colors_sorted_by_depth,
    std::vector<double> &opacities_sorted_by_depth,
    std::vector<Eigen::Vector2d> &centers_sorted_by_depth,
    std::vector<double> &depths_sorted,
    const std::vector<Eigen::Vector3d> &pks,
    const std::vector<Eigen::Vector3d> &scales,
    const std::vector<Eigen::Vector4d> &quats,
    const std::vector<Eigen::Vector3d> &colors,
    const std::vector<double> &opacities,
    const Eigen::Matrix4d &viewmat,
    const Eigen::Matrix<double, 3, 4> &projmat
    ) {
    size_t N = pks.size();

    std::vector<Eigen::Matrix3d> transforms(N);
    std::vector<Eigen::Vector2d> centers(N);
    std::vector<double> depths(N);
    for (size_t idx = 0; idx < N; idx++) {
        Eigen::Matrix3d rotation = BuildScalingRotation(scales[idx], quats[idx]);

        // 1. Viewing transform
        // # Eq.4 and Eq.5
        Eigen::Matrix<double, 4, 3> H = Eigen::Matrix<double, 4, 3>::Zero();
        H.block<3, 2>(0, 0) = rotation.block<3, 2>(0, 0);  // 取前两列
        H.block<3, 1>(0, 2) = pks[idx];
        H(3, 2) = 1.0;

        Eigen::Matrix<double, 4, 3> VH = viewmat * H;
        Eigen::Matrix3d WH = projmat * VH;

        Eigen::Vector2d center_point, extent;
        ComputeAABB(center_point, extent, WH);

        transforms[idx] = WH;
        centers[idx] = center_point;
        depths[idx] = VH(2, 2);
    }

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](int i, int j) {return depths[i] < depths[j];});

    Ts.resize(N);
    colors_sorted_by_depth.resize(N);
    opacities_sorted_by_depth.resize(N);
    centers_sorted_by_depth.resize(N);
    depths_sorted.resize(N);
    for (size_t idx = 0; idx < N; idx++) {
        Ts[idx] = transforms[indices[idx]];
        colors_sorted_by_depth[idx] = colors[indices[idx]];
        opacities_sorted_by_depth[idx] = opacities[indices[idx]];
        centers_sorted_by_depth[idx] = centers[indices[idx]];
        depths_sorted[idx] = depths[indices[idx]];
    }

}


void AlphaBlendingWithGaussians(
    Eigen::Vector3d &color,
    double &depth,
    const std::vector<double> &dist2s,
    const std::vector<Eigen::Vector3d> &colors,
    const std::vector<double> &opacities,
    const std::vector<double> &depths,
    const size_t &H,
    const size_t &W
    ) {
    // init values
    color = Eigen::Vector3d(0, 0, 0);
    depth = 0;

    double cutoff = std::pow(1.0, 2);
    size_t N = dist2s.size();
    double transmittance = 1.0;
    double cumulate_weight = 0.0;
    for (size_t idx = 0; idx < N; idx++) {
        double dist2 = dist2s[idx];
        double opacity = opacities[idx];

        // Eq. 3.
        // Obtain alpha by multiplying with Gaussian opacity
        // and its exponential falloff from mean.
        // Avoid numerical instabilities (see paper appendix).
        double gaussian = dist2 < cutoff ? std::exp(-0.5 * dist2) : 0;
        double alpha = std::min(0.99, opacity * gaussian);
        if (alpha < 1.0 / 255.0) {
            continue;
        }

        float test_transmittance = transmittance * (1 - alpha);
        if (test_transmittance < 0.0001) {
            break;
        }

        float weight = alpha * transmittance;
        color += colors[idx] * weight;
        depth += depths[idx] * weight;

        transmittance = test_transmittance;
        cumulate_weight += weight;
    }

    if (cumulate_weight > 0.001)
        depth /= cumulate_weight;
}


void SurfaceSplating(
    cv::Mat &image,
    cv::Mat &depthmap,
    const std::vector<Eigen::Vector3d> &pks,
    const std::vector<Eigen::Vector3d> &scales,
    const std::vector<Eigen::Vector4d> &quats,
    const std::vector<Eigen::Vector3d> &colors,
    const std::vector<double> &opacities,
    const Eigen::Matrix3d &intrins,
    const Eigen::Matrix4d &viewmat,
    const size_t image_width,
    const size_t image_height
    ) {
    size_t N = pks.size();

    Eigen::Matrix<double, 3, 4> projmat = Eigen::Matrix<double, 3, 4>::Zero();
    projmat.block<3, 3>(0, 0) = intrins;

    // Rasterization setup
    std::vector<Eigen::Matrix3d> Ts;
    std::vector<Eigen::Vector3d> colors_sorted_by_depth;
    std::vector<double> opacities_sorted_by_depth;
    std::vector<Eigen::Vector2d> centers_sorted_by_depth;
    std::vector<double> depths_sorted;
    Setup(Ts, colors_sorted_by_depth, opacities_sorted_by_depth, centers_sorted_by_depth, depths_sorted,
          pks, scales, quats, colors, opacities, viewmat, projmat);

    // Rasterization

    // 1. Generate pixels
    size_t H = image_height;
    size_t W = image_width;
    image = cv::Mat (H, W, CV_32FC3);
    depthmap = cv::Mat(H, W, CV_32FC1);

    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            Eigen::Vector3d hx(-1, 0, col);
            Eigen::Vector3d hy(0, -1, row);

            std::vector<double> dist2s(N), depths_acc(N);
            for (size_t idx = 0; idx < N; idx++) {
                const auto &T = Ts[idx];

                // 2. Compute ray splat intersection # Eq.9 and Eq.10
                Eigen::Vector3d hu = T.transpose() * hx;
                Eigen::Vector3d hv = T.transpose() * hy;

                double u = (hu[1]*hv[2] - hu[2]*hv[1])/(hu[0]*hv[1] - hu[1]*hv[0]);
                double v = (hu[2]*hv[0] - hu[0]*hv[2])/(hu[0]*hv[1] - hu[1]*hv[0]);

                // 3. add low pass filter # Eq. 11
                // when a point (2D Gaussian) viewed from a far distance or from a slended angle
                // the 2D Gaussian will falls between pixels and no fragment is used to rasterize the Gaussian
                // so we should add a low pass filter to handle such aliasing.
                double dist_3d = u*u + v*v;
                double filter_size = std::sqrt(2.0) / 2;
                Eigen::Vector2d delt_2d = Eigen::Vector2d(u, v) - centers_sorted_by_depth[idx];
                double dist_2d = (1.0/filter_size)*(1.0/filter_size) * (delt_2d[0]*delt_2d[0] + delt_2d[1]*delt_2d[1]);

                double dist, depth_acc;
                if (dist_3d < dist_2d) {
                    dist = dist_3d;
                    depth_acc = T(2, 0) * u + T(2, 1) * v + T(2, 2);
                } else {
                    dist = dist_2d;
                    depth_acc = T(2, 2);
                }
                dist2s[idx] = dist;
                depths_acc[idx] = depth_acc;
            }

            Eigen::Vector3d color;
            double depth;
            AlphaBlendingWithGaussians(color, depth, dist2s, colors_sorted_by_depth, opacities_sorted_by_depth, depths_acc, H, W);

            image.ptr<cv::Vec3f>(row)[col] = cv::Vec3f(color[0], color[1], color[2]);
            depthmap.ptr<float>(row)[col] = depth;
        }
    }
}

int main(int argc, char** argv) {
    size_t num_points = 8;

    std::vector<Eigen::Vector3d> pks, scales;
    std::vector<Eigen::Vector4d> quats;

    GetInputs(pks, scales, quats, num_points);

    Eigen::Matrix3d intrins;
    Eigen::Matrix4d viewmat, projmat;
    size_t width, height;
    GetCameras(intrins, viewmat, projmat, width, height);

    std::vector<Eigen::Vector3d> colors(pks.size());
    for (auto &color : colors) {
        color = Eigen::Vector3d(rand(), rand(), rand()) / RAND_MAX;
    }

    std::vector<double> opacities(pks.size(), 1.0);

    cv::Mat image, depthmap;
    SurfaceSplating(image, depthmap, pks, scales, quats, colors, opacities, intrins, viewmat, width, height);
    std::cout << "done" << std::endl;

    return 0;
}

