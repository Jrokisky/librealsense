// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#pragma once

#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "synthetic-stream.h"

namespace rs2 
{
	class stream_profile;
}

namespace librealsense {

    class ransac_filter : public stream_filter_processing_block
    {
    public:
        static std::shared_ptr<ransac_filter> create();

        ransac_filter();

        virtual void depth_to_points(
		float * points, 
		const uint16_t * depth_image, 
		const rs2_intrinsics &depth_intrinsics,
		float depth_scale);

    protected:
        rs2::frame process_frame(const rs2::frame_source& source, const rs2::frame& f) override;

    private:
        void highlight_plane(const bool plane_found, const bool* inliers, uint16_t* depth_data, uint16_t* new_data, int size);
        float4 generate_equation(const float3* points, int size);
    	int get_inliers(const float4& eq, const float3* points, int size, bool* inliers, float threshold);

        rs2::stream_profile     _source_stream_profile;
        rs2::stream_profile     _target_stream_profile;
	rs2_intrinsics         _depth_intrinsics;
        float                  _depth_units;
	float4		       _equation;
	bool  		       _plane_found;
	float		       _iterations;
	float		       _distance_threshold;
	float   	       _threshold_percent;

    };
    MAP_EXTENSION(RS2_EXTENSION_RANSAC_FILTER, librealsense::ransac_filter);

}
