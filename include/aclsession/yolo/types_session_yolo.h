#pragma once
#include <vector>
#include <variant>
#include "types_session_core.h"

namespace AI::CV::YOLO
{
    /// @brief YOLO 执行 LetterBox 时保存原始参数的结构体
    struct LetterBoxMeta
    {
        int imOriW{0};
        int imOriH{0};
        float scale{1.0f};
        int left{0};
        int top{0};
        int right{0};
        int bottom{0};
    };

    /// @brief YOLO Session 基础输入结构
    struct BaseIn
    {
        std::unique_ptr<std::vector<std::vector<float>>> data{nullptr};
    };

    /// @brief YOLO Session 采用 AIPP 时的输入结构
    struct AippIn
    {
        std::unique_ptr<std::vector<std::vector<uint8_t>>> data{nullptr};
    };

    /// @brief YOLO Session 采用 DVPP 时的输入结构
    struct DvppIn
    {
        std::unique_ptr<std::vector<std::vector<uint8_t>>> data{nullptr};
        std::unique_ptr<std::vector<LetterBoxMeta>> meta{nullptr};

        // Auxiliary Data
        int imwidth{640}, imheight{640}, lettpad{114};

        // 记录 Batch 中为每条 Bin 数据预留 reserve 的内存空间大小 (初始值为 2*1024*1024 即2MB)
        std::unique_ptr<std::vector<int>> curSz;
    };

    /// @brief YOLO Session 基础输出结构
    struct BaseOut
    {
        std::unique_ptr<std::vector<std::vector<float>>> data{nullptr};
    };

    /// @brief YOLO Session 采用 NMS 算子时的输出结构
    struct NmsOpOut
    {
        std::unique_ptr<std::vector<std::vector<float>>> data{nullptr};
        std::unique_ptr<std::vector<int32_t>> indices{nullptr};

        // Auxiliary Data
        int clsnum{80}, anchorsnum{-1}, maxdetspercla{15};
        float confth{0.25}, iouth{0.7};

        std::vector<int> boxStride;
        std::vector<int> scoreStride;
    };

    using BaseParam = SessionParamExec<BaseIn, BaseOut>;
    using AippParam = SessionParamExec<AippIn, BaseOut>;
    using DvppParam = SessionParamExec<DvppIn, BaseOut>;
    using BaseNmsParam = SessionParamExec<BaseIn, NmsOpOut>;
    using AippNmsParam = SessionParamExec<AippIn, NmsOpOut>;
    using DvppNmsParam = SessionParamExec<DvppIn, NmsOpOut>;

    using YOLOParam = std::variant<
        BaseParam,
        AippParam,
        DvppParam,
        BaseNmsParam,
        AippNmsParam,
        DvppNmsParam>;

    struct TimeInfoSessionExtra
    {
        float ld{0.0f};
        float let{0.0f};
        float nms{0.0f};

        void Init(){*this={};}
    };
}