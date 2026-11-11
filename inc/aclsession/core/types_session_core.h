#pragma once

#include <memory>

namespace AI
{
    /// @brief ACL Session 执行推理会话时所需的输入，输出参数以及辅助参数
    /// @tparam In  模型推理输入数据类型
    /// @tparam Out  模型推理输出数据类型
    template <typename In, typename Out>
    struct SessionParamExec
    {
        using InType        = In;
        using OutType       = Out;

        In dataIn;
        Out dataOut;
    };


    /// @brief Session 推理基础耗时记录体
    struct TimeInfoSessionCore
    {
        float h2d{0.0f};
        float inf{0.0f};
        float d2h{0.0f};

        void Init(){*this = {};}
    };
}