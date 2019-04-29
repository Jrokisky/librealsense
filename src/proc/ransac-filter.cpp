// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include "../include/librealsense2/rs.hpp"
#include "../include/librealsense2/rsutil.h"

#include "proc/ransac-filter.h"
#include "option.h"
#include "context.h"
#include "environment.h"
#include "threshold.h"
#include "image.h"

#ifdef RS2_USE_CUDA
#include "proc/cuda/cuda-ransac-filter.h"
#endif

namespace librealsense {

    // The number of iterations to attempt to find a plane with valid inliers.
    const float iterations_min = 1.0f;
    const float iterations_max = 50.0f;
    const float iterations_default = 25.0f;
    const float iterations_step = 1.0f;

    // The distance that a point can be from the plane to be considered an inlier.
    const float distance_threshold_min = 0.0;
    const float distance_threshold_max = 0.25;
    const float distance_threshold_default = 0.05;
    const float distance_threshold_step = 0.01;

    // The minimum percent of pixels in the scene that must be inliers to be a valid solution.
    const float threshold_percent_min = 1.0f;
    const float threshold_percent_max = 100.0f;
    const float threshold_percent_default = 30.0f;
    const float threshold_percent_step = 1.0f;

    std::shared_ptr<ransac_filter> ransac_filter::create() {
        #ifdef RS2_USE_CUDA
	    return std::make_shared<librealsense::ransac_filter_cuda>();
        #else
	    return std::make_shared<librealsense::ransac_filter>();
	#endif
    }

    ransac_filter::ransac_filter() : 
	    stream_filter_processing_block("Ransac Filter"),
	    _plane_found(false),
	    _iterations(iterations_default),
	    _distance_threshold(distance_threshold_default),
	    _threshold_percent(threshold_percent_default)
    {

        _stream_filter.format = RS2_FORMAT_Z16;
        _stream_filter.stream = RS2_STREAM_DEPTH;

        auto iterations_control = std::make_shared<ptr_option<float>>(
            iterations_min,
            iterations_max,
            iterations_step,
            iterations_default,
            &_iterations, "RANSAC Iterations");
	register_option(static_cast<rs2_option>(RS2_OPTION_FILTER_MAGNITUDE), iterations_control);

        auto distance_threshold_control = std::make_shared<ptr_option<float>>(
            distance_threshold_min,
            distance_threshold_max,
            distance_threshold_step,
            distance_threshold_default,
            &_distance_threshold, "RANSAC Distance Threshold");
	register_option(static_cast<rs2_option>(RS2_OPTION_MIN_DISTANCE), distance_threshold_control);

        auto threshold_percent_control = std::make_shared<ptr_option<float>>(
            threshold_percent_min,
            threshold_percent_max,
            threshold_percent_step,
            threshold_percent_default,
            &_threshold_percent, "RANSAC Threshold Percent");
	register_option(static_cast<rs2_option>(RS2_OPTION_MAX_DISTANCE), threshold_percent_control);
    }

    /**
     * This can be parallelized.
     */
    void ransac_filter::depth_to_points(float * points, const uint16_t * depth_image, const rs2_intrinsics &depth_intrinsics, float depth_scale)
    {
 	for (int y = 0; y < depth_intrinsics.height; ++y)
        {
            for (int x = 0; x < depth_intrinsics.width; ++x)
            {
                const float pixel[] = { (float)x, (float)y };
                rs2_deproject_pixel_to_point(points, &depth_intrinsics, pixel, depth_scale * (*depth_image));
		depth_image++;
                points += 3;
            }
        }
    }
    
    /**
     * This can be parallelized.
     */
    int ransac_filter::get_inliers(const float4& eq, const float3* points, int size, bool* inliers, float threshold) {
	int inlier_count = 0;
	// Loop through our points.
        for (int i = 0; i < size; i++) {
            float3 curr_point = points[i];

	    // Holes in the image are considered outliers.
	    if (curr_point.z < 0.01f) {
		inliers[i] = false;
	    }
	    else {
		// Compute distance between point and plane equation using the formula
	        // at https://mathinsight.org/distance_point_plane.
	        float numerator = fabs((eq.x * curr_point.x) + (eq.y * curr_point.y) + (eq.z * curr_point.z) + eq.w);
	        float denominator = sqrtf((eq.x * eq.x) + (eq.y * eq.y) + (eq.z * eq.z));
		if ( denominator > 0.01f) {
	            float distance = numerator / denominator;
                    if (distance < threshold) {
	                inlier_count++;
	                inliers[i] = true;
	            }
	            else {
	                inliers[i] = false;
	            }  
		} else {
	            inliers[i] = false;
	        }
	    }
	}
	return inlier_count;
    }

    void ransac_filter::highlight_plane(const bool* inliers, uint16_t* depth_data, uint16_t* new_data, int size) {
	for (int i = 0; i < size; i++) {
	    if (inliers[i] ) {
                new_data[i] = depth_data[i];
	    }
	}
    }

    void ransac_filter::generate_equation(float4 *equation, const float3* points, int size) {
	float3 vector_ab;
	float3 vector_ac;
	float3 point_a;
	// Ensure that the 3 points we're using for the plane generation are 
	// not colinnear and their depths are not 0.
	bool invalid = true;
	do { 
	    // Select 3 unique random points.
	    int a = 0, b = 0, c = 0;
	    while (a == b || b == c || a == c) {
	        a = rand() % (size-1);
	        b = rand() % (size-1);
	        c = rand() % (size-1);
	    }

	    point_a = points[a];
	    float3 point_b = points[b];
	    float3 point_c = points[c];

	    // Don't use holes in the image with depth of 0.0.
	    if (point_a.z < 0.01f) continue;
	    if (point_b.z < 0.01f) continue;
	    if (point_c.z < 0.01f) continue;

	    // Compute our vectors.
	    vector_ab = point_b - point_a;
	    vector_ac = point_c - point_a;

	    // Ensure that the three points are not collinear by ensuring the 
	    // vectors are not parallel. Parallel vectors will have a similar ratio
	    // beween their components. Ex: vectora = 4 * vectorb (they are parallel)
	    float ratio_x = vector_ab.x / vector_ac.x;
	    float ratio_y = vector_ab.y / vector_ac.y;
	    float ratio_z = vector_ab.z / vector_ac.z;
	    bool equal_xy = fabs(ratio_x - ratio_y) < 0.01;
	    bool equal_yz = fabs(ratio_y - ratio_z) < 0.01;
	    bool equal_xz = fabs(ratio_x - ratio_z) < 0.01;
	    invalid = equal_xy && equal_yz && equal_xz;
	} while(invalid);

	// Compute the cross product of the vectors.
	float cpx = vector_ab.y * vector_ac.z - vector_ab.z * vector_ac.y;
	float cpy = vector_ab.z * vector_ac.x - vector_ab.x * vector_ac.z;
	float cpz = vector_ab.x * vector_ac.y - vector_ab.y * vector_ac.x;

	// Use the cross product and point a to find the constant in the plane equation.
	float d = -(cpx * point_a.x + cpy * point_a.y + cpz * point_a.z);

        equation->x = cpx;
        equation->y =cpy;
        equation->z = cpz;
        equation->w = d;
    }

    void ransac_filter::run_ransac(bool *inliers, const uint16_t * depth_data, int size, const rs2_intrinsics &depth_intrinsics, float depth_units)
    {
        // Initialize random seed.
	srand((unsigned)time(0));

	float3* points = new float3[size];
        // Conver the depth data into 3d space.
        ransac_filter::depth_to_points(reinterpret_cast<float *>(points), depth_data, depth_intrinsics, depth_units);
	
        // RANSAC settings.
	int inlier_threshold_count = (((int)_threshold_percent) * size) / 100;
	   
	for (int j = 0; j < (int)_iterations; j++) {
	    // Generate a random plane equation, if our last equation did not find a plane.
	    if (!_plane_found) {
                generate_equation(&_equation, points, size);
	    }

	    // Get the inliers & count using this equation.
	    int inlier_count = get_inliers(_equation, points, size, inliers, _distance_threshold);
	    if (inlier_count >= inlier_threshold_count) {
                _plane_found = true;
                break;
            }
            else {
                _plane_found = false;
	    }
	}
    }

    rs2::frame ransac_filter::process_frame(const rs2::frame_source& source, const rs2::frame& f)
    {
	if (!f.is<rs2::depth_frame>()) return f;

        if (f.get_profile().get() != _source_stream_profile.get())
        {
            _source_stream_profile = f.get_profile();
            _target_stream_profile = f.get_profile().clone(RS2_STREAM_DEPTH, 0, RS2_FORMAT_Z16);
        }

	auto vf = f.as<rs2::depth_frame>();
        auto width = vf.get_width();
        auto height = vf.get_height();
        auto new_f = source.allocate_video_frame(_target_stream_profile, f,
            vf.get_bytes_per_pixel(), width, height, vf.get_stride_in_bytes(), RS2_EXTENSION_DEPTH_FRAME);
        auto stream_profile = f.get_profile();
        auto video = stream_profile.as<rs2::video_stream_profile>();
          
        if (new_f && video)
        {
            auto ptr = dynamic_cast<librealsense::depth_frame*>((librealsense::frame_interface*)new_f.get());
            auto orig = dynamic_cast<librealsense::depth_frame*>((librealsense::frame_interface*)f.get());

            auto depth_data = (uint16_t*)orig->get_frame_data();
            auto new_data = (uint16_t*)ptr->get_frame_data();

            // Get the intrinsics of the image.
            _depth_intrinsics = video.get_intrinsics();
            ptr->set_sensor(orig->get_sensor());
            auto _depth_units = orig->get_units();
            memset(new_data, 0, width * height * sizeof(uint16_t));

	    // Pixel area of the depth image.
	    int size = width * height;

	    // Stores if a pixel is an inlier or not.
	    bool* inliers = new bool[size];

            // Run Ransac.
            run_ransac(inliers, depth_data, size, _depth_intrinsics, _depth_units); 
            
	    // Highlight our ground plane.
 	    highlight_plane(inliers, depth_data, new_data, size);

            // Free memory.
	    delete[] inliers;
	    return new_f;
	}
	return f;
    }
    
}
