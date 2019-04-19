// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include "../include/librealsense2/rs.hpp"
#include "../include/librealsense2/rsutil.h"

#include "proc/ransac-filter.h"

namespace librealsense {

    std::shared_ptr<ransac_filter> ransac_filter::create() {
        #ifdef RS2_USE_CUDA
	    // TODO: update this. Based on cuda-pointcloud.
	    return std::make_shared<librealsense::ransac_filter>();
        #else
	    return std::make_shared<librealsense::ransac_filter>();
	#endif
    }

    ransac_filter::ransac_filter()
        :stream_filter_processing_block("Ransac Filter")
    {
	_plane_found = false;
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

    void ransac_filter::highlight_plane(const bool plane_found, const bool* inliers, rs2::video_frame rgb, int size) {
        auto rgb_data = reinterpret_cast<uint8_t*>(const_cast<void *>(rgb.get_data()));
	for (int i = 0; i < size; i++) {
	    if (inliers[i] ) {
	        // Neon green for inliers.
                rgb_data[i * 3 + 0] = 57;
                rgb_data[i * 3 + 1] = 255;
                rgb_data[i * 3 + 2] = 20;
	    } else {
                rgb_data[i * 3 + 0] = 255;
	        // Red for outliers.
                rgb_data[i * 3 + 1] = 0;
                rgb_data[i * 3 + 2] = 0;;
	    }
	}
    }

    float4 ransac_filter::generate_equation(const float3* points, int size) {
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

	return float4 {cpx, cpy, cpz, d};
    }

    rs2::frame ransac_filter::process_frame(const rs2::frame_source& source, const rs2::frame& f)
    {
	rs2::frame ret;
	if (f.is<rs2::depth_frame>())
	{

	   // Get information about frame.
	   auto stream_profile = f.get_profile();
	   if (auto video = stream_profile.as<rs2::video_stream_profile>())
	   {
	       // Get the intrinsics of the image.
	       _depth_intrinsics = video.get_intrinsics();
	       auto sensor = ((frame_interface*)f.get())->get_sensor().get();
	       // Depth resolution.
               _depth_units = sensor->get_option(RS2_OPTION_DEPTH_UNITS).query();
	       // Depth of each pixel.
               auto depth_data = (const uint16_t*)f.get_data();
	       // Pixel area of the depth image.
	       int size = _depth_intrinsics.height * _depth_intrinsics.width;
	       float3* points = new float3[size];
	       // Stores if a pixel is an inlier or not.
	       bool* inliers = new bool[size];
	       // Initialize random seed.
	       srand((unsigned)time(0));

	       // Conver the depth data into 3d space.
               ransac_filter::depth_to_points(reinterpret_cast<float *>(points), depth_data, _depth_intrinsics, _depth_units);
	
	       // RANSAC settings.
	       int iterations = 25;
	       float distance_threshold = 0.03;
	       int inlier_threshold = (3 * size) / 10;
	       
	       for (int j = 0; j < iterations; j++) {
		   // Generate a random plane equation, if our last equation did not find a plane.
	           if (!_plane_found) {
	               _equation = generate_equation(points, size);
	           }

		   // Get the inliers & count using this equation.
		   int inlier_count = get_inliers(_equation, points, size, inliers, distance_threshold);
		   if (inlier_count >= inlier_threshold) {
			   _plane_found = true;
			   break;
		   }
		   else {
			   _plane_found = false;
	           }
	       }

	       // Create our output rgb frame.
               auto vf = f.as<rs2::video_frame>();
	       auto target_stream_profile = stream_profile.clone(RS2_STREAM_DEPTH, 0, RS2_FORMAT_RGB8);
               ret = source.allocate_video_frame(target_stream_profile, f, 3, vf.get_width(), vf.get_height(), vf.get_width() * 3, RS2_EXTENSION_VIDEO_FRAME);

	       // Highlight our ground plane.
 	       highlight_plane(_plane_found, inliers, ret, size);

	       // Free memory.
	       delete[] points;
	       delete[] inliers;
	    }
	}
	return ret;
    }

}
