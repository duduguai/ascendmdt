#pragma once

#include "models_core_yolo.h"

namespace AI::CV::YOLO
{
    /// @brief YOLO 模型基类
    class YOLO
    {
        protected:
            /// @brief 模型核心
            std::unique_ptr<BaseCore> core;
        
        public:
            YOLO()=default;
            virtual ~YOLO();

            /// @brief YOLO 系列模型初始化方法
            /// @return 是否成功初始化
            bool Init();

            /// @brief YOLO 系列模型推理执行入口
            /// @param input 推理输入源
            /// @param saveDir 推理输出绘制图像保存目录
            /// @param verbose 是否显示推理各阶段耗时信息
            /// @return 是否执行成功
            bool Execute(const std::string& input, std::optional<std::string> saveDir=std::nullopt, bool verbose=true);

            /// @brief YOLO 系列模型推理执行入口
            /// @param imBatchPaths 推理输入源
            /// @param outputs 推理输出, 最多 BatchSize 个 OutputPerIm, OutputPerIm 可在 commontypes_yolo.h 中查看
            /// @param verbose true: 显示推理信息, false: 推理静默 
            /// @return 是否执行成功
            bool Execute(std::vector<std::filesystem::path> imBatchPaths, std::vector<OutputPerIm>& outputs, bool verbose=true);
    };
}