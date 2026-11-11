#pragma once
#include <memory>
#include <optional>
#include <filesystem>

#include "opencv2/opencv.hpp"
#include "aclsession_yolo.h"
#include "types_session_yolo.h"
#include "types_models_yolo.h"
#include "commontypes_yolo.h"
namespace AI::CV::YOLO
{
    
    /// @brief 非DVPP下模型处理输入需要的参数
    struct BaseCoeForIn
    {   
        int imWidth;
        int imHeight;
        int numsPerChannel;
        int numsPerIm;
        int lettpad;

        std::vector<cv::Mat> imVecMat;
        std::vector<LetterBoxMeta> meta;
    };

    /// @brief 非NMS算子下模型处理输出需要的参数
    struct BaseCoeForOut
    {
        int clsNum; 
        int scoreNum;
        int anchorsNum;
        int maxDets;
        float confTh;
        float iouTh;
    };

    /// @brief 部署YOLO分割模型推理需要的参数
    struct SegCoe
    {
        int protoWidth;
        int protoHeight;
        int numsPerProto;
        int numsPerMask;
        int masknum{32};
        float maskTh{0.5f};
    };

    /// @brief YOLO 模型 Core 的基底
    class BaseCore
    {
        public:
            // 模型名称 
            std::string name{""};
            // 模型ID
            int id{-1};

        protected:
        //// 固有参数
            // 创建实例统计量
            inline static int _count = 0;

            // 记录耗时信息
            std::unique_ptr<TimeInfoYOLO> timeInfo{nullptr};

            bool isYOLOX{false};
            int batchSize{1};
            int boxCordsNum{4};

            // // DVPP 默认为 Batch 中读取的每条二进制数据 reserve 保留的内存空间大小 (2MB)
            // int defauSzDvpp{2*1024*1024};

            // Session 执行参数
            YOLOParam param;
            // ACL 会话
            std::unique_ptr<AclSessionYOLO> session{nullptr};

            // 通过路径解析得到的图像文件集
            std::vector<std::filesystem::path> imPaths;
            // 加载有效图像信息记录
            std::vector<int> imValidIdx;

            // 保存类别信息向量
            std::vector<std::string> catsName;

            // 输入, 输出处理参数
            std::unique_ptr<BaseCoeForIn> coeIn;
            std::unique_ptr<BaseCoeForOut> coeOut;
            std::unique_ptr<SegCoe> coeSeg;

        //// 基类与派生类共用方法
            /// @brief 预留解码与NMS过滤接口
            /// @param ptr 解码图像对应推理输出数据起始指针
            /// @param outBoxes 存放输出数据的容器
            virtual void PostDecode(const float* ptr, std::vector<BoundingBox>& outBoxes);

            /// @brief 预留 Seg 模型解码与NMS过滤接口
            /// @param detPtr 检测任务输出数据起始指针
            /// @param segPtr 分割任务输出数据起始指针
            /// @param boxes 存放解码输出的 BoundingBox 容器
            /// @param masks 存放解码输出的 Mask 容器
            virtual void PostDecode(const float* detPtr, const float* segPtr, std::vector<BoundingBox>& boxes,
                        std::vector<cv::Mat>& masks);
        
        private:
        //// 私有方法
            /// @brief 将输入路径解析为图像文件路径向量 
            /// @param input 输入路径 (图像文件路径 or 保存图像的目录)
            /// @return 是否解析成功
            bool CollectFiles(const std::string& input);

            /// @brief 获取推理完成需要的 Epoch 总数
            /// @return Epoch 总数
            int GetEpoches();

            /// @brief 通过 OpenCV 加载并解码图像文件
            /// @param offset 当前 Batch 起始位相对于所有数据的起始位偏移
            /// @param validNum 加载后, 当前 Batch 中加载的实际有效数量
            /// @return 是否加载成功 (当实际有效数量为 0 时表示加载失败)
            bool LoadBatchImMat(int offset, int& validNum);

            /// @brief 通过加载二进制数据完成图像读取 (用于后续DVPP解码)
            /// @param imBinBatch 保存读取的二进制数据容器
            /// @param recordSz 记录 Batch 中每条 Bin 数据在内存中预留的内存大小
            /// @param offset 当前 Batch 起始位相对于所有数据的起始位偏移
            /// @param validNum 加载后, 当前 Batch 中加载的实际有效数量
            /// @return 是否加载成功 (当实际有效数量为 0 时表示加载失败)
            bool LoadBatchImBin(std::vector<std::vector<uint8_t>>& imBinBatch, std::vector<int>& recordSz, int offset, int& validNum);

            /// @brief 通过OpenCV实现LetterBox逻辑, 单次调用完成单张图像的处理
            /// @param imIn 输入图像
            /// @param ImOut 输出图像
            /// @param meta 记录 LetterBox 缩放和填充参数
            void LetterBoxViaOpenCV(const cv::Mat& imIn, cv::Mat& ImOut, LetterBoxMeta& meta);

            /// @brief 实现归一化 1) uint8 -> float; 2) BGR -> RGB; 3) /255.0, 单次调用处理单张图像
            /// @param imIn 输入图像
            /// @param dataOut 保存输出结果的容器
            /// @param idx 处理图像位于当前 Batch 的序号
            void Normalize(const cv::Mat& imIn, std::vector<float>& dataOut, int idx);

            /// @brief 针对使用 NMS 算子时, 模型后续解码操作
            /// @param boxesPtr 解码图像对应推理输出 boxes 数据起始指针
            /// @param scoresPtr 解码图像对应推理输出 scores 数据起始指针
            /// @param indicesPtr 解码图像对应推理输出 indices 数据起始指针
            /// @param boxStride boxes 的数据跨越步长
            /// @param scoreStride scores 的数据跨越步长
            /// @param outBoxes 保存输出结果的容器
            /// @param numsPerData 单张图像推理输出的数据量
            /// @param idx 处理图像位于当前 Batch 的序号
            void PostDecode(const float* boxesPtr, const float* scoresPtr, const int32_t* indicesPtr,
                std::vector<int>& boxStride, std::vector<int>& scoreStride,
                std::vector<BoundingBox>& outBoxes, int numsPerData, int idx);
            
            /// @brief 将推理输出检测框缩放回输入图像尺寸上
            /// @param output 单张图像的推理输出
            /// @param meta LetterBox 时的处理参数
            void BBoxToOri(OutputPerIm& output, const LetterBoxMeta& meta);
            
            /// @brief 将推理输出掩码缩放回输入图像尺寸上
            /// @param output 单张图像的推理输出
            /// @param meta LetterBox 时的处理参数
            /// @param imwidth 输入进模型的图像宽
            /// @param imheight 输入进模型的图像高
            /// @return 是否成功
            bool MaskToOri(OutputPerIm& output, const LetterBoxMeta& meta, int imwidth, int imheight);
            
            /// @brief 将推理输出检测狂绘制在图像上
            /// @param im 输入原始图像
            /// @param output 单张图像的推理输出
            void DrawBoundingBox(cv::Mat& im, const OutputPerIm& output);


        public:
            BaseCore(int mode, const std::string& modelpath, int deviceid, int batchsize, int imwidth, int imheight, float confTh, 
                bool recordtime, int maxdets, int clsnum, int lettpad, std::optional<std::vector<std::string>> catsname=std::nullopt,
                std::optional<bool> enableseg=std::nullopt);
            virtual ~BaseCore();
            
            /// @brief 模型内核初始化方法
            /// @return 是否初始化成功
            bool Init();
            
            /// @brief YOLO 模型推理执行入口, 该入口有输出. 单次调用该入口只处理 BatchSize 个图像文件, 输出 BatchSize 个 OutputPerIm.
            /// @param imBatchPaths 输入的 BatchSize 个图像文件路径
            /// @param output 输出的 BatchSize 个OutputPerIm
            /// @param verbose true: 显示推理信息, false: 推理静默 
            /// @return 是否执行成功
            bool Execute(std::vector<std::filesystem::path> imBatchPaths, std::vector<OutputPerIm>& outputs, bool verbose);

            /// @brief YOLO 模型推理执行入口, 该入口无输出. 若给出了 saveDir, 该接口会将推理结果绘图并保存到指定目录中
            /// @param input 模型推理的输入来源 (图像文件路径 or 保存图像的目录)
            /// @param saveDir 保存输出推理绘图的目录
            /// @param verbose 是否显示推理汇总信息
            /// @return 是否执行成功
            bool Execute(const std::string& input, std::optional<std::string> saveDir=std::nullopt, bool verbose=true);
    };

    /// @brief AnchorBase 类 YOLO 模型, 输出格式形如 [BatchSize, AnchorsNum, BoxCordsNum+1+ClsNum]
    class AnchorBasedCore : public BaseCore
    {
        private:
            int gridsnumWS, gridsnumHS, gridsnumWM, gridsnumHM, gridsnumWL, gridsnumHL;

            /// @brief YOLOX 解码时获取 anchor 对应的横向与纵向 grid, 以及 stride
            /// @param anchorIdx anchor 序号
            /// @param gridx 返回的横向 grid 序号
            /// @param gridy 返回的纵向 grid 序号
            /// @param stride 返回的 stride
            void GetGridStride(int anchorIdx, int& gridx, int& gridy, int& stride);

        protected:
            void PostDecode(const float* ptr, std::vector<BoundingBox>& outBoxes) override;

        public:
            AnchorBasedCore(int mode, const std::string& modelpath, std::string modelname, int deviceid, int batchsize, int imwidth, int imheight, 
                float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad, int anchorsfactor, 
                std::optional<std::vector<std::string>> catsname=std::nullopt, bool isYolox=false);
            
            ~AnchorBasedCore();
            
    };

    /// @brief AnchorFree 类 YOLO 模型, 输出格式形如 [BatchSize, BoxCordsNum+ClsNum, AnchorsNum]
    class AnchorFreeCore : public BaseCore
    {
        protected:
            void PostDecode(const float* ptr, std::vector<BoundingBox>& outBoxes) override;

            void PostDecode(const float* detPtr, const float* segPtr, std::vector<BoundingBox>& boxes,
                            std::vector<cv::Mat>& masks) override;
        
        public:
            AnchorFreeCore(int mode, const std::string& modelpath, std::string modelname, int deviceid, int batchsize, int imwidth, int imheight,
                float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad, 
                std::optional<std::vector<std::string>> catsname=std::nullopt,
                std::optional<bool> enableseg=std::nullopt);

            ~AnchorFreeCore();
    };

    /// @brief End2End 类 YOLO 模型, 输出格式形如 [BatchSize, 300, 6]
    class EndToEndCore : public BaseCore
    {
        protected:
            void PostDecode(const float* ptr, std::vector<BoundingBox>& outBoxes) override;

            void PostDecode(const float* detPtr, const float* segPtr, std::vector<BoundingBox>& boxes,
                            std::vector<cv::Mat>& masks) override;
        
        public:
            EndToEndCore(int mode, const std::string& modelpath, std::string modelname, int deviceid, int batchsize, int imwidth, int imheight,
                float confTh, int anchorsnum, bool recordtime, int maxdets, int lettpad, std::optional<std::vector<std::string>> catsname=std::nullopt,
                std::optional<bool> enableseg=std::nullopt);

            ~EndToEndCore();
    };
}