#ifdef RS2_USE_CUDA

#include "cuda-ransac-filter.cuh"
#include <iostream>
#include "../../include/librealsense2/rsutil.h"
#include "../types.h"
#include "rscuda_utils.cuh"

// Pulled from Librealsense.
__device__ static void rs2_deproject_pixel_to_point_cuda(float point[3], const struct rs2_intrinsics * intrin, const float pixel[2], float depth)
{
    assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
    assert(intrin->model != RS2_DISTORTION_FTHETA); // Cannot deproject to an ftheta image

    float x = (pixel[0] - intrin->ppx) / intrin->fx;
    float y = (pixel[1] - intrin->ppy) / intrin->fy;

    if (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
    {
        float r2 = x * x + y * y;
        float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2*r2 + intrin->coeffs[4] * r2*r2*r2;
        float ux = x * f + 2 * intrin->coeffs[2] * x*y + intrin->coeffs[3] * (r2 + 2 * x*x);
        float uy = y * f + 2 * intrin->coeffs[3] * x*y + intrin->coeffs[2] * (r2 + 2 * y*y);
        x = ux;
        y = uy;
    }
    point[0] = depth * x;
    point[1] = depth * y;
    point[2] = depth;
}

// Pulled from Librealsense.
__global__
void rs2_kernel_deproject_depth_cuda(float * points, const rs2_intrinsics* intrin, const uint16_t * depth, float depth_scale)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i >= (*intrin).height * (*intrin).width) {
        return;
    }
    int stride = blockDim.x * gridDim.x;
    int a, b;
    
    for (int j = i; j < (*intrin).height * (*intrin).width; j += stride) {
        b = j / (*intrin).width;
        a = j - b * (*intrin).width;
        const float pixel[] = { (float)a, (float)b };
        rs2_deproject_pixel_to_point_cuda(points + j * 3, intrin, pixel, depth_scale * depth[j]);
    }    
}

void depth_pixel_to_point(
    librealsense::float3 *point, 
    const rs2_intrinsics &depth_intrinsics, 
    int location,
    float depth)
{
    // Convert location to x, y coordinates.
    int width = depth_intrinsics.width;
    const float pixel[] = { (float) (location % width), (float) (location / width) };
    rs2_deproject_pixel_to_point(reinterpret_cast<float *>(point), &depth_intrinsics, pixel, depth);
}

/**
 * Unlike with the serial version, we directly use the depth data (not points) as
 * we want to avoid the overhead of copying the points from the device to host.
 **/
void rscuda::generate_equation(
    const uint16_t *depth_data, 
    librealsense::float4 *equation, 
    int size,
    const rs2_intrinsics &depth_intrinsics, 
    float depth_scale)
{
    librealsense::float3 vector_ab = {};
    librealsense::float3 vector_ac = {};
    librealsense::float3 point_a = {};
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

        // For each of our points, get their location in 3d space.
        depth_pixel_to_point(&point_a, depth_intrinsics, a, depth_scale * depth_data[a]);
	librealsense::float3 point_b = {};
        depth_pixel_to_point(&point_b, depth_intrinsics, b, depth_scale * depth_data[b]);
	librealsense::float3 point_c = {};
        depth_pixel_to_point(&point_c, depth_intrinsics, c, depth_scale * depth_data[c]);

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
    equation->y = cpy;
    equation->z = cpz; 
    equation->w = d;
}

__global__
void get_inliers(
    const float4 *dev_equation, 
    const float3 *dev_points, 
    const int *dev_size, 
    bool *dev_inliers, 
    const float *dev_distance_threshold)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int j = i; j < (*dev_size); j += stride) {
	float3 curr_point = dev_points[i];
	// Holes in the image are considered outliers.
	if (curr_point.z < 0.01f) {
            dev_inliers[j] = false;
	}
	else {
            // Compute distance between point and plane equation using the formula
	    // at https://mathinsight.org/distance_point_plane.
	    float numerator = fabs((dev_equation->x * curr_point.x) + (dev_equation->y * curr_point.y) + (dev_equation->z * curr_point.z) + dev_equation->w);
	    float denominator = sqrtf((dev_equation->x * dev_equation->x) + (dev_equation->y * dev_equation->y) + (dev_equation->z * dev_equation->z));
	    if ( denominator > 0.01f) {
	        float distance = numerator / denominator;
                if (distance < (*dev_distance_threshold)) {
	            dev_inliers[j] = true;
	        }
	        else {
	            dev_inliers[j] = false;
	        }  
            } else {
	        dev_inliers[j] = false;
	    }
	}
    }
}

void rscuda::ransac_filter_cuda(
    bool *inliers, 
    const uint16_t * depth_data, 
    int size, 
    const rs2_intrinsics &depth_intrinsics, 
    float *depth_scale, 
    bool *plane_found, 
    librealsense::float4 *equation, 
    const float distance_threshold,
    const float threshold_percent,
    const float iterations)
{
    // Initialize random seed.
    srand((unsigned)time(0));

    // RANSAC settings.
    int inlier_threshold_count = (((int)threshold_percent) * size) / 100;

    // CUDA KERNEL DEPTH_TO_POINTS
    int count = depth_intrinsics.height * depth_intrinsics.width;
    int numBlocks = count / RS2_CUDA_THREADS_PER_BLOCK;

    // Declare device variables.
    float3 *dev_points = 0;
    uint16_t *dev_depth_data = 0;
    rs2_intrinsics* dev_intrin = 0;
    float4 *dev_equation = 0;
    int *dev_size = 0;
    bool *dev_inliers = 0;
    float *dev_distance_threshold = 0;

    cudaError_t result;

    // Allocate Memory on the Device.
    result = cudaMalloc(&dev_points, count * sizeof(float3));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_inliers, count * sizeof(bool));
    assert(result == cudaSuccess);

    result = cudaMalloc(&dev_depth_data, count * sizeof(uint16_t));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_intrin, sizeof(rs2_intrinsics));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_equation, sizeof(float4));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_size, sizeof(int));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_distance_threshold, sizeof(float));
    assert(result == cudaSuccess);

    // Copy values over to cuda device.
    result = cudaMemcpy(dev_depth_data, depth_data, count * sizeof(uint16_t), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess); 
    result = cudaMemcpy(dev_intrin, &depth_intrinsics, sizeof(rs2_intrinsics), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess); 
    result = cudaMemcpy(dev_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess); 
    result = cudaMemcpy(dev_distance_threshold, &distance_threshold, sizeof(float), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess); 
 
    // Convert depth image to points.
    rs2_kernel_deproject_depth_cuda<<<numBlocks, RS2_CUDA_THREADS_PER_BLOCK>>>(reinterpret_cast<float *>(dev_points), dev_intrin, dev_depth_data, *depth_scale);
    cudaDeviceSynchronize();
    bool * best_inliers;
    best_inliers = new bool[size];
    int prev_best = 0;

    for (int j = 0; j < (int)iterations; j++) {
        // Generate a random plane equation, if our last equation did not find a plane.
	if (!(*plane_found)) {
            generate_equation(depth_data, equation, size, depth_intrinsics, *depth_scale);
	}
	result = cudaMemcpy(dev_equation, equation, sizeof(float4), cudaMemcpyHostToDevice);
	assert(result == cudaSuccess); 

        // Get the inliers.
	get_inliers<<<numBlocks, RS2_CUDA_THREADS_PER_BLOCK>>>(dev_equation, dev_points, dev_size, dev_inliers, dev_distance_threshold);
        cudaDeviceSynchronize();

        // Copy inliers from device to host
     	result = cudaMemcpy(inliers, dev_inliers, count * sizeof(bool), cudaMemcpyDeviceToHost);
     	assert(result == cudaSuccess);

        // loop through array of inliers and get a count.
        int inlier_count = 0;
        for (int i = 0; i < size; i++) {
            if (inliers[i]) inlier_count++;
            best_inliers[i] = inliers[i];
        }
	if (inlier_count >= inlier_threshold_count) {
            //LOG_WARNING("=======================================================================plane_found");
            (*plane_found) = true;
            break;
        }
        else {
            //LOG_WARNING("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$plane_not_found");
            prev_best = inlier_count;
            (*plane_found) = false;
	}
    }

    // If no plane found, use the best solution we did find.
    if (!(*plane_found)) {
        for (int i = 0; i < size; i++) {
            inliers[i] = best_inliers[i];
        }
    }

    // Free memory on CUDA.
    cudaFree(dev_points);
    cudaFree(dev_depth_data);
    cudaFree(dev_intrin);
    cudaFree(dev_equation);
    cudaFree(dev_size);
    cudaFree(dev_inliers);
    cudaFree(dev_distance_threshold);
}

#endif
