#pragma once
#include "opencv2/opencv.hpp"

namespace AI::CV::YOLO
{
    struct TimeInfoYOLO
    {
        float ld{0.0f};
        float let{0.0f};
        float nor{0.0f};
        float nms{0.0f};
        float sto{0.0f};

        void Init(){ *this = {};}
    };

    struct OriMask
    {
        std::vector<float> data;
        
        OriMask() = default;

        OriMask(int size) : data(size, 0.0f){}

        cv::Mat DataToMat(int protowidth, int protoheight)
        {
            cv::Mat maskMat(protoheight, protowidth, CV_32F, data.data());
            maskMat = maskMat.clone();
            return maskMat;
        }
    };
}