#pragma once
#include <vector>
#include <memory>

#include "common_types.h"

namespace ZWXT::AI{

    struct BaseParam{
        // 输入部分
        std::unique_ptr<std::vector<std::vector<float>>> dataInBase{nullptr};
        // 输出部分
        std::unique_ptr<std::vector<std::vector<float>>> dataOutBase{nullptr};
    };

    namespace CV{

        struct LetterBoxMeta{
            int imOriW{0};
            int imOriH{0};
            float scale{1.0f};
            int left{0};
            int top{0};
            int right{0};
            int bottom{0};
        };

        struct AippParam{
            std::unique_ptr<std::vector<std::vector<uint8_t>>> dataInAipp{nullptr};
        };

        struct DvppParam{
            std::unique_ptr<std::vector<std::vector<uint8_t>>> dataInDvpp{nullptr};
            std::unique_ptr<std::vector<LetterBoxMeta>> letterboxMeta{nullptr};
        };
        
        namespace YOLO{
            struct NmsOpParam{
                std::unique_ptr<std::vector<float>> boxes;
                std::unique_ptr<std::vector<float>> scores;
                std::unique_ptr<std::vector<int32_t>> indices;
            };
        }
    }

    struct BatchTimeInfo{
        // 总览部分
        float pre{0.0f};
        float e2e{0.0f};
        float pos{0.0f};

        // Session 部分
        float h2d{0.0f};
        float inf{0.0f};
        float d2h{0.0f};

        // CV部分
        float ld{0.0f};
        float let{0.0f};
        float nor{0.0f};
        float nms{0.0f};
        float sto{0.0f};

        void Init(){ *this = {}; }
    };

    struct SessionExecParam{
        BaseParam baseParam;

        // CV Param
        CV::AippParam aippParam;
        CV::DvppParam dvppParam;

        // CV DET Param
        CV::YOLO::NmsOpParam nmsopParam;
        std::unique_ptr<BatchTimeInfo> timeInfo{nullptr};
    };

}