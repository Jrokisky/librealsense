#pragma once
#ifndef LIBREALSENSE_CUDA_RANSAC_FILTER_H
#define LIBREALSENSE_CUDA_RANSAC_FILTER_H

#ifdef RS2_USE_CUDA

// Types
#include <stdint.h>
#include "../../include/librealsense2/rs.h"
#include "assert.h"
#include "../../include/librealsense2/rsutil.h"
#include <functional>
#include "../types.h"

// CUDA headers
#include <cuda_runtime.h>

#ifdef _MSC_VER 
// Add library dependencies if using VS
#pragma comment(lib, "cudart_static")
#endif

#define RS2_CUDA_THREADS_PER_BLOCK 256

namespace rscuda
{

    void rscuda::ransac_filter_cuda(
    	bool *inliers, 
    	const uint16_t * depth_data, 
    	librealsense::float3 *points, 
    	int size, 
    	const rs2_intrinsics &depth_intrinsics, 
    	float *depth_scale, 
    	bool &plane_found, 
    	librealsense::float4 &equation, 
    	const float distance_threshold,
    	const float threshold_percent,
    	const float iterations);

    void rscuda::generate_equation(
    	const uint16_t *depth_data, 
    	librealsense::float4 *equation, 
    	int size,
    	const rs2_intrinsics &depth_intrinsics, 
    	float depth_scale);


}

#endif // RS2_USE_CUDA

#endif // LIBREALSENSE_CUDA_RANSAC_FILTER_H
