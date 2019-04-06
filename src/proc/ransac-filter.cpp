// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include "../include/librealsense2/rs.hpp"

#include "proc/ransac-filter.h"

namespace librealsense {

    std::shared_ptr<ransac_filter> ransac_filter::create() {
        #ifdef RS2_USE_CUDA
	    return std::make_shared<librealsense::ransac_filter>();
        #else
	    return std::make_shared<librealsense::ransac_filter>();
	#endif
    }

    ransac_filter::ransac_filter()
        :stream_filter_processing_block("Ransac Filter")
    {

    }

    rs2::frame ransac_filter::process_frame(const rs2::frame_source& source, const rs2::frame& f)
    {
	    return f;
    }

}
