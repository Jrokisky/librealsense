// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#pragma once
#include "../include/librealsense2/hpp/rs_frame.hpp"
#include "synthetic-stream.h"

namespace librealsense {

    class LRS_EXTENSION_API ransac_filter : public stream_filter_processing_block
    {
    public:
        static std::shared_ptr<ransac_filter> create();

        ransac_filter();

    protected:
        rs2::frame process_frame(const rs2::frame_source& source, const rs2::frame& f) override;

    };

}
