#pragma once

#include <memory>
#include <vector>

namespace AI
{
    /// @brief Session 模型推理输入 Base 类型
    struct BaseIn
    {
        std::unique_ptr<std::vector<std::vector<float>>> data;
    };
    
    /// @brief Session 模型推理输出 Base 类型
    struct BaseOut
    {
        std::unique_ptr<std::vector<std::vector<float>>> data;
    };
}