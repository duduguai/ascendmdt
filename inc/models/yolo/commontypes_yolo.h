#pragma once
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

namespace AI::CV::YOLO
{
    struct BoundingBox
    {
        float x1{0.0f};
        float y1{0.0f};
        float x2{0.0f};
        float y2{0.0f};
        float score{0.0f};
        int cls{-1};
    };

    struct Detection
    {
        std::vector<BoundingBox> boxes;
    };

    struct Segmantation
    {
        std::vector<cv::Mat> masks;
    };

    struct OutputPerIm
    {
        std::string filename;

        Detection det;
        std::unique_ptr<Segmantation> seg;
    };
}