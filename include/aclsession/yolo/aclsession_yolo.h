#pragma once
#define ENABLE_DVPP_INTERFACE
#include <acl/ops/acl_dvpp.h>
#include <aclnnop/aclnn_ops_cv.h>

#include "aclsession_core.h"
#include "types_session_yolo.h"

namespace AI::CV::YOLO
{
    struct DvppCoeffi
    {
        // LetterBox 填充像素值
        double lettPad{114};

        // 模型输入图像尺寸信息
        int imWidth{640};
        int imHeight{640};

        // DVPP 通道描述符
        acldvppChannelDesc* JPEGD{nullptr};
        acldvppChannelDesc* VPC{nullptr};

        // 预设 binary 图像默认大小 (Bytes)
        // int defaultBinSz{2*1024*1024};
        std::unique_ptr<std::vector<int>> defaultBinSz;
        std::unique_ptr<std::vector<void*>> imsBinVecPtr{nullptr};
        // 预设经 DVPP 解码后图像默认大小 (Bytes)
        // int defaultDecSz{4*1024*1024};
        std::unique_ptr<std::vector<int>> defaultDecSz;
        std::unique_ptr<std::vector<void*>> imsDecVecPtr{nullptr};

        // 使用 DVPP 进行 LetterBox 所需操作描述符
        acldvppBatchPicDesc* srcBtPicDesc{nullptr};
        acldvppBatchPicDesc* dstBtPicDesc{nullptr};
        std::unique_ptr<std::vector<uint32_t>> roiNumsPtr{nullptr};
        std::unique_ptr<std::vector<acldvppRoiConfig*>> cropAreasPtr{nullptr};
        std::unique_ptr<std::vector<acldvppBorderConfig*>> borderCfgsPtr{nullptr};
        acldvppResizeConfig* resizeCfg{nullptr};
    };

    struct NmsOpCoeffi
    {
        /// 使用 NMS OP 需要引入的变量
        // 构建时输入变量
        int clsNum{-1}, anchorsNum{-1};
        int64_t maxDetsPerCla{-1};
        float confTh{0.25f}, iouTh{0.7f};

        // NMS OP 运行所需参数
        aclTensor* boxes{nullptr};
        aclTensor* scores{nullptr};
        aclTensor* indices{nullptr};
        void* indicesAddr{nullptr};
        size_t indicesDevSz{0};

        aclIntArray* maxDetsPerClaIntArr{nullptr};
        aclFloatArray* iouThFloArr{nullptr};
        aclFloatArray* confThFloArr{nullptr};
        
        uint64_t workSpaceSz{0};
        void* workSpaceAddr{nullptr};

        aclOpExecutor* executor{nullptr};
    };

    class AclSessionYOLO : public AclSession
    {
        private:
            //// 参数
            // SessionExecParam
            YOLOParam* paramPtr{nullptr};

            // YOLO Session 补充记录阶段耗时
            std::unique_ptr<TimeInfoSessionExtra> timeInfoExtra{nullptr};

            // YOLO Session DVPP and NMS 算子专用参数
            std::unique_ptr<DvppCoeffi> dvpp{nullptr};
            std::unique_ptr<NmsOpCoeffi> nms{nullptr};

            //// 内部方法
            /// 使用 DVPP 需要引入的方法
            /// @brief 创建 DVPP 操作通道
            /// @param desc DVPP 操作通道描述符
            /// @param mode DVPP 操作通道的类型, 如 (JPEGD 或 VPC)
            /// @param modename DVPP 操作通道的名称, 如 ("JPEGD" 或 "VPC")
            /// @return 是否创建成功
            bool CreateDvppChannel(acldvppChannelDesc*& desc, acldvppChannelMode mode, std::string modename);

            /// @brief 通过 DVPP 对 JPEG 图像完成解码操作
            /// @return 是否解码成功
            bool DvppDecode();

            /// @brief 通过 DVPP 实现 LetterBox 操作
            /// @return 是否成功
            bool DvppLetterBox();


            /// 使用 NMS 算子需要引入的方法
            /// @brief NMS 算子执行
            /// @return 是否执行成功
            bool NmsOpExecute();

            //// 需要重载的函数
            bool CheckTensorConsist(int requiredNum, int mode) override;

            bool CheckBytesConsist(int bufferSz, int mode, int idx) override;

            bool PrepareBuffer(int idx, void*& buffer, int bufferSz, int mode) override;

            bool Host2DeviceEP() override;

            bool Device2HostEP() override;

            bool PreProcess() override;

            bool PostProcess() override;

        public:
            AclSessionYOLO(const std::string& modelpath, int deviceid, YOLOParam* paramptr, int batchsize, bool needtimeinfo);
            
            ~AclSessionYOLO();

            bool Init() override;
            
            /// @brief RC模式下，使用 DVPP 时, 当读取的二进制文件大于默认大小时，更新内存索引
            /// @param addr 更新的内存索引
            /// @param index Batch 中的序号
            void DvppApdateAddr(void* addr, int index);

            /// @brief 当使用DVPP或NMS算子时, 返回NPU上执行ld, let, nms各阶段的耗时
            /// @return 单次推理耗时记录
            TimeInfoSessionExtra GetTimeInfoExtra();
    };

}