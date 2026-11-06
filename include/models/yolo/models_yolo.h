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


    /// @brief YOLOv5
    class YOLOv5: public YOLO
    {
        public:
            /// @brief YOLOv5 构造函数
            /// @param mode       模式
            /// @param modelpath  om 模型文件路径 
            /// @param deviceid   部署的 NPU 设备号
            /// @param batchsize  om 模型的 BatchSize 配置
            /// @param catsname   推理检测的类别名向量
            /// @param imwidth    模型文件配置的输入数据宽度
            /// @param imheight   模型文件配置的输入数据高度
            /// @param confTh     置信度阈值
            /// @param iouTh      交并比阈值
            /// @param recordtime 是否记录推理运行各阶段耗时
            /// @param maxdets    单张图像推理输出最大保留框数量
            /// @param clsnum     类别数量
            /// @param lettpad    LetterBox 填充时采用的像素值
            YOLOv5(int mode, const std::string& modelpath, int deviceid, int batchsize=1, std::optional<std::vector<std::string>> catsname=std::nullopt,
                int imwidth=640, int imheight=640, float confTh=0.25f, float iouTh=0.7f, bool recordtime=true, int maxdets=300, int clsnum=80, 
                int lettpad=114);
            ~YOLOv5(){}
    };

    /// @brief YOLOv5u
    class YOLOv5u: public YOLO
    {
        public:
            /// @brief YOLOv5u 构造函数
            /// @param mode       模式
            /// @param modelpath  om 模型文件路径 
            /// @param deviceid   部署的 NPU 设备号
            /// @param batchsize  om 模型的 BatchSize 配置
            /// @param catsname   推理检测的类别名向量
            /// @param imwidth    模型文件配置的输入数据宽度
            /// @param imheight   模型文件配置的输入数据高度
            /// @param confTh     置信度阈值
            /// @param iouTh      交并比阈值
            /// @param recordtime 是否记录推理运行各阶段耗时
            /// @param maxdets    单张图像推理输出最大保留框数量
            /// @param clsnum     类别数量
            /// @param lettpad    LetterBox 填充时采用的像素值
            YOLOv5u(int mode, const std::string& modelpath, int deviceid, int batchsize=1, std::optional<std::vector<std::string>> catsname=std::nullopt,
                int imwidth=640, int imheight=640, float confTh=0.25f, float iouTh=0.7f, bool recordtime=true, int maxdets=300, int clsnum=80, 
                int lettpad=114);
            ~YOLOv5u(){}
    };

    /// @brief YOLOv6
    class YOLOv6: public YOLO
    {
        public:
            /// @brief YOLOv6 构造函数
            /// @param mode       模式
            /// @param modelpath  om 模型文件路径 
            /// @param deviceid   部署的 NPU 设备号
            /// @param batchsize  om 模型的 BatchSize 配置
            /// @param catsname   推理检测的类别名向量
            /// @param imwidth    模型文件配置的输入数据宽度
            /// @param imheight   模型文件配置的输入数据高度
            /// @param confTh     置信度阈值
            /// @param iouTh      交并比阈值
            /// @param recordtime 是否记录推理运行各阶段耗时
            /// @param maxdets    单张图像推理输出最大保留框数量
            /// @param clsnum     类别数量
            /// @param lettpad    LetterBox 填充时采用的像素值
            YOLOv6(int mode, const std::string& modelpath, int deviceid, int batchsize=1, std::optional<std::vector<std::string>> catsname=std::nullopt,
                int imwidth=640, int imheight=640, float confTh=0.25f, float iouTh=0.7f, bool recordtime=true, int maxdets=300, int clsnum=80, 
                int lettpad=114);
            ~YOLOv6(){}
    };

    /// @brief YOLOv7
    class YOLOv7: public YOLO
    {
        public:
            /// @brief YOLOv7 构造函数
            /// @param mode       模式
            /// @param modelpath  om 模型文件路径 
            /// @param deviceid   部署的 NPU 设备号
            /// @param batchsize  om 模型的 BatchSize 配置
            /// @param catsname   推理检测的类别名向量
            /// @param imwidth    模型文件配置的输入数据宽度
            /// @param imheight   模型文件配置的输入数据高度
            /// @param confTh     置信度阈值
            /// @param iouTh      交并比阈值
            /// @param recordtime 是否记录推理运行各阶段耗时
            /// @param maxdets    单张图像推理输出最大保留框数量
            /// @param clsnum     类别数量
            /// @param lettpad    LetterBox 填充时采用的像素值 
            YOLOv7(int mode, const std::string& modelpath, int deviceid, int batchsize=1, std::optional<std::vector<std::string>> catsname=std::nullopt,
                int imwidth=640, int imheight=640, float confTh=0.25f, float iouTh=0.7f, bool recordtime=true, int maxdets=300, int clsnum=80, 
                int lettpad=114);
            ~YOLOv7(){}
    };

    /// @brief YOLOv8
    class YOLOv8: public YOLO
    {
        public:
            /// @brief YOLOv8 构造函数
            /// @param mode       模式
            /// @param modelpath  om 模型文件路径 
            /// @param deviceid   部署的 NPU 设备号
            /// @param batchsize  om 模型的 BatchSize 配置
            /// @param catsname   推理检测的类别名向量
            /// @param imwidth    模型文件配置的输入数据宽度
            /// @param imheight   模型文件配置的输入数据高度
            /// @param confTh     置信度阈值
            /// @param iouTh      交并比阈值
            /// @param recordtime 是否记录推理运行各阶段耗时
            /// @param maxdets    单张图像推理输出最大保留框数量
            /// @param clsnum     类别数量
            /// @param lettpad    LetterBox 填充时采用的像素值
            YOLOv8(int mode, const std::string& modelpath, int deviceid, int batchsize=1, std::optional<std::vector<std::string>> catsname=std::nullopt,
                bool isSeg=false, int imwidth=640, int imheight=640, float confTh=0.25f, float iouTh=0.7f, bool recordtime=true, int maxdets=300, 
                int clsnum=80, int lettpad=114);
            ~YOLOv8(){}
    };

    /// @brief YOLOv9
    class YOLOv9: public YOLO
    {
        public:
            /// @brief YOLOv9 构造函数
            /// @param mode       模式
            /// @param modelpath  om 模型文件路径 
            /// @param deviceid   部署的 NPU 设备号
            /// @param batchsize  om 模型的 BatchSize 配置
            /// @param catsname   推理检测的类别名向量
            /// @param imwidth    模型文件配置的输入数据宽度
            /// @param imheight   模型文件配置的输入数据高度
            /// @param confTh     置信度阈值
            /// @param iouTh      交并比阈值
            /// @param recordtime 是否记录推理运行各阶段耗时
            /// @param maxdets    单张图像推理输出最大保留框数量
            /// @param clsnum     类别数量
            /// @param lettpad    LetterBox 填充时采用的像素值
            YOLOv9(int mode, const std::string& modelpath, int deviceid, int batchsize=1, std::optional<std::vector<std::string>> catsname=std::nullopt,
                int imwidth=640, int imheight=640, float confTh=0.25f, float iouTh=0.7f, bool recordtime=true, int maxdets=300, int clsnum=80, 
                int lettpad=114);
            ~YOLOv9(){}
    };

    /// @brief YOLO10
    class YOLO10: public YOLO
    {
        public:
            /// @brief YOLO10 构造函数
            /// @param mode         模式
            /// @param modelpath    om 模型文件路径 
            /// @param deviceid     部署的 NPU 设备号
            /// @param batchsize    om 模型的 BatchSize 配置
            /// @param catsname     推理检测的类别名向量
            /// @param imwidth      模型文件配置的输入数据宽度
            /// @param imheight     模型文件配置的输入数据高度
            /// @param confTh       置信度阈值
            /// @param anchorsnum   单张图像推理输出的 Anchor 数量
            /// @param recordtime   是否记录推理运行各阶段耗时
            /// @param maxdets      单张图像推理输出最大保留框数量
            /// @param lettpad      LetterBox 填充时采用的像素值
            YOLO10(int mode, const std::string& modelpath, int deviceid, int batchsize=1, std::optional<std::vector<std::string>> catsname=std::nullopt,
                int imwidth=640, int imheight=640, float confTh=0.25f, int anchorsnum=300, bool recordtime=true, int maxdets=300, int lettpad=114);
            ~YOLO10(){}
    };

    /// @brief YOLO11
    class YOLO11: public YOLO
    {
        public:
            /// @brief YOLO11 构造函数
            /// @param mode       模式
            /// @param modelpath  om 模型文件路径 
            /// @param deviceid   部署的 NPU 设备号
            /// @param batchsize  om 模型的 BatchSize 配置
            /// @param catsname   推理检测的类别名向量
            /// @param imwidth    模型文件配置的输入数据宽度
            /// @param imheight   模型文件配置的输入数据高度
            /// @param confTh     置信度阈值
            /// @param iouTh      交并比阈值
            /// @param recordtime 是否记录推理运行各阶段耗时
            /// @param maxdets    单张图像推理输出最大保留框数量
            /// @param clsnum     类别数量
            /// @param lettpad    LetterBox 填充时采用的像素值
            YOLO11(int mode, const std::string& modelpath, int deviceid, int batchsize=1, std::optional<std::vector<std::string>> catsname=std::nullopt,
                bool isSeg=false, int imwidth=640, int imheight=640, float confTh=0.25f, float iouTh=0.7f, bool recordtime=true, int maxdets=300, 
                int clsnum=80, int lettpad=114);
            ~YOLO11(){}
    };

    /// @brief YOLO12
    class YOLO12: public YOLO
    {
        public:
            /// @brief YOLO12 构造函数
            /// @param mode       模式
            /// @param modelpath  om 模型文件路径 
            /// @param deviceid   部署的 NPU 设备号
            /// @param batchsize  om 模型的 BatchSize 配置
            /// @param catsname   推理检测的类别名向量
            /// @param imwidth    模型文件配置的输入数据宽度
            /// @param imheight   模型文件配置的输入数据高度
            /// @param confTh     置信度阈值
            /// @param iouTh      交并比阈值
            /// @param recordtime 是否记录推理运行各阶段耗时
            /// @param maxdets    单张图像推理输出最大保留框数量
            /// @param clsnum     类别数量
            /// @param lettpad    LetterBox 填充时采用的像素值
            YOLO12(int mode, const std::string& modelpath, int deviceid, int batchsize=1, std::optional<std::vector<std::string>> catsname=std::nullopt,
                int imwidth=640, int imheight=640, float confTh=0.25f, float iouTh=0.7f, bool recordtime=true, int maxdets=300, int clsnum=80, 
                int lettpad=114);
            ~YOLO12(){}
    };

    /// @brief YOLO13
    class YOLO13: public YOLO
    {
        public:
            /// @brief YOLO13 构造函数
            /// @param mode       模式
            /// @param modelpath  om 模型文件路径 
            /// @param deviceid   部署的 NPU 设备号
            /// @param batchsize  om 模型的 BatchSize 配置
            /// @param catsname   推理检测的类别名向量
            /// @param imwidth    模型文件配置的输入数据宽度
            /// @param imheight   模型文件配置的输入数据高度
            /// @param confTh     置信度阈值
            /// @param iouTh      交并比阈值
            /// @param recordtime 是否记录推理运行各阶段耗时
            /// @param maxdets    单张图像推理输出最大保留框数量
            /// @param clsnum     类别数量
            /// @param lettpad    LetterBox 填充时采用的像素值
            YOLO13(int mode, const std::string& modelpath, int deviceid, int batchsize=1, std::optional<std::vector<std::string>> catsname=std::nullopt,
                int imwidth=640, int imheight=640, float confTh=0.25f, float iouTh=0.7f, bool recordtime=true, int maxdets=300, int clsnum=80, 
                int lettpad=114);
            ~YOLO13(){}  
    };

    /// @brief YOLO26
    class YOLO26: public YOLO
    {
        public:
            /// @brief YOLO26 构造函数
            /// @param mode         模式
            /// @param modelpath    om 模型文件路径 
            /// @param deviceid     部署的 NPU 设备号
            /// @param batchsize    om 模型的 BatchSize 配置
            /// @param catsname     推理检测的类别名向量
            /// @param imwidth      模型文件配置的输入数据宽度
            /// @param imheight     模型文件配置的输入数据高度
            /// @param confTh       置信度阈值
            /// @param anchorsnum   单张图像推理输出的 Anchor 数量
            /// @param recordtime   是否记录推理运行各阶段耗时
            /// @param maxdets      单张图像推理输出最大保留框数量
            /// @param lettpad      LetterBox 填充时采用的像素值
            YOLO26(int mode, const std::string& modelpath, int deviceid, int batchsize=1, std::optional<std::vector<std::string>> catsname=std::nullopt,
                bool isSeg=false, int imwidth=640, int imheight=640, float confTh=0.25f, int anchorsnum=300, bool recordtime=true, int maxdets=300, 
                int lettpad=114);
            ~YOLO26(){}
    };

    /// @brief YOLOX
    class YOLOX: public YOLO
    {
        public:
            /// @brief YOLOX 构造函数
            /// @param mode       模式
            /// @param modelpath  om 模型文件路径 
            /// @param deviceid   部署的 NPU 设备号
            /// @param batchsize  om 模型的 BatchSize 配置
            /// @param catsname   推理检测的类别名向量
            /// @param imwidth    模型文件配置的输入数据宽度
            /// @param imheight   模型文件配置的输入数据高度
            /// @param confTh     置信度阈值
            /// @param iouTh      交并比阈值
            /// @param recordtime 是否记录推理运行各阶段耗时
            /// @param maxdets    单张图像推理输出最大保留框数量
            /// @param clsnum     类别数量
            /// @param lettpad    LetterBox 填充时采用的像素值
            YOLOX(int mode, const std::string& modelpath, int deviceid, int batchsize=1, std::optional<std::vector<std::string>> catsname=std::nullopt,
                int imwidth=640, int imheight=640, float confTh=0.25f, float iouTh=0.7f, bool recordtime=true, int maxdets=300, 
                int clsnum=80, int lettpad=114);
            ~YOLOX(){}
    };
}