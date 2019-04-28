// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#pragma once
#include "../ransac-filter.h"

namespace librealsense
{
    class ransac_filter_cuda : public ransac_filter
    {
    public:
        ransac_filter_cuda();
    private:
        void run_ransac(bool *inliers, const uint16_t * depth_image, float3 *points, int size, const rs2_intrinsics &depth_intrinsics, float depth_scale);

    };
}
