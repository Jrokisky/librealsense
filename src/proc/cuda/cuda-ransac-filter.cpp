// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.
#include "proc/cuda/cuda-ransac-filter.h"

#ifdef RS2_USE_CUDA
#include "../../cuda/cuda-ransac-filter.cuh"
#endif

namespace librealsense
{
    ransac_filter_cuda::ransac_filter_cuda() : ransac_filter() {}

    void ransac_filter_cuda::run_ransac(bool *inliers, const uint16_t * depth_image, int size, const rs2_intrinsics &depth_intrinsics, float depth_scale)
    {
#ifdef RS2_USE_CUDA
        bool plane_found = _plane_found;
        float4 equation = {
            _equation.x,
            _equation.y,
            _equation.z,
            _equation.w
        };
        rscuda::ransac_filter_cuda(inliers, depth_image, size, depth_intrinsics, &depth_scale, &plane_found, &equation, _distance_threshold, _threshold_percent, _iterations);
        
        _plane_found = plane_found;

        _equation.x = equation.x;
        _equation.y = equation.y;
        _equation.z = equation.z;
        _equation.w = equation.w;

#endif
    }


}
