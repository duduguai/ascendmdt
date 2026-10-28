#pragma once

namespace ZWXT::AI{
    namespace CV{
        namespace YOLO{
            struct BoundingBox{
                float x1{0.0f};
                float y1{0.0f};
                float x2{0.0f};
                float y2{0.0f};
                float score{0.0f};
                int cls{-1};
            };

            struct Detection{
                std::string filename;
                std::vector<BoundingBox> box;
            };
        }
    }
}