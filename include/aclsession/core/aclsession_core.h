#pragma once
#include <string>
#include <vector>
#include <memory>
#include <acl/acl.h>

#include "types_session_core.h"
namespace AI
{
    /// @brief ioType 对照表 {0: input, 1: output}
    const std::vector<std::string> ioType{"Input", "Output"};

    /// @brief 维护模型输入输出信息结构体; buffer -> 保存数据的首地址指针; size -> 保存数据的大小, 单位 Byte; dataBuffer -> ACL 用于维护数据块信息的结构体
    struct datainfo{
        void* buffer;
        std::size_t size;
        aclDataBuffer* dataBuffer;
    };

    /// @brief AclSession 推理会话核心, 以此为基础向外派生
    class AclSession
    {
        protected:
            // 实例创建计数
            inline static int _count = 0;
            // ACL 运行环境变量
            inline static aclrtRunMode runMode=ACL_HOST;
            aclrtContext context{nullptr};
            aclrtStream stream{nullptr};

            // ACL 错误码
            aclError ret;

            // 模型变量
            std::string modelPath;
            int deviceId{0};
            aclmdlDesc* modelDesc{nullptr};
            uint32_t modelId{0};
            void* modelMemBuffer{nullptr};
            void* modelWeightBuffer{nullptr};
            size_t modelMemSize{0};
            size_t modelWeightSize{0};

            // 模型输入输出变量
            std::vector<datainfo> inputDataInfos;
            std::vector<datainfo> outputDataInfos;
            aclmdlDataset* inputDataset{nullptr};
            aclmdlDataset* outputDataset{nullptr};

            // 信息变量
            int batchSize{1};

            // 是否记录各阶段耗时
            std::unique_ptr<TimeInfoSessionCore> timeInfoPtr{nullptr};

            /// @brief 判断发送给 NPU 的指令是否成功下发
            /// @param OperType 操作的名称
            /// @param coutFlg 当错误时是否将错误信息打印到终端
            /// @return 是否成功下发
            bool check(const std::string& OperType="", bool coutFlg=false);


            /// @brief 初始化 ACL 运行环境, 并完成 Model Load
            /// @return 是否完成初始化
            bool BaseInit();


            /// @brief 检查 SessionExecParam 中的 Tensor 数量是否与模型要求的一致
            /// @param requireNum 模型要求的 Tensor 数量 
            /// @param mode 输入 or 输出 {0: Input, 1: Output}
            /// @return 是否满足要求
            virtual bool CheckTensorConsist(int requireNum, int mode);


            /// @brief 检查 SessionExecParam 中各 Tensor 的数据量是否与模型要求的一致
            /// @param bufferSz 模型要求的数据量 Bytes
            /// @param mode 输入 or 输出 {0: Input, 1: Output}
            /// @param idx Tensor 的序号
            /// @return 是否满足要求
            virtual bool CheckBytesConsist(int bufferSz, int mode, int idx);

            
            /// @brief Session 模型构造输入, 输出数据的内存空间
            /// @param idx 输入或输出数据的序号
            /// @param buffer 创建或绑定的内存空间首地址
            /// @param bufferSz 需要的内存空间大小
            /// @param mode 0->输入; 1->输出
            virtual bool PrepareBuffer(int idx, void*& buffer, int bufferSz, int mode);
            

            /// @brief 为 Session 中的模型创建输入, 输出的描述符; 同时构建或绑定内存空间
            /// @param ioMode 输入 or 输出 {0: Input, 1: Output}
            /// @param modelDesc 模型信息描述符
            /// @param datainfos 模型输入, 输出信息维护数据体
            /// @param dataset 模型输入, 输出描述符
            /// @return 是否成功构建
            bool PrepareDateset(int ioMode, aclmdlDesc* modelDesc, std::vector<datainfo>& datainfos,
                aclmdlDataset*& dataset);
            

            /// @brief EP 模式下, 将 Host 侧数据转移至 Device 侧
            /// @return 是否成功转移
            virtual bool Host2DeviceEP();


            /// @brief EP 模式下, 将 Device 侧数据转移至 Host 侧
            /// @return 是否成功转移
            virtual bool Device2HostEP();
            
            /// @brief Session 推理前处理
            /// @return 前处理是否成功
            virtual bool PreProcess();


            /// @brief Session 推理任务挂载
            /// @return 推理是否成功
            bool Inference();

            
            /// @brief Session 推理后处理
            /// @return 后处理是否成功
            virtual bool PostProcess();

        public:
            /// @brief 构造函数
            /// @param modelpath 模型文件路径 
            /// @param deviceid 部署使用的 NPU ID
            /// @param batchsize 部署推理使用的 BatchSize
            /// @param needrecordtime 是否记录推理过程中 h2d, inf, d2h 三个阶段的耗时
            AclSession(const std::string& modelpath, int deviceid, int batchsize, bool needrecordtime);
            virtual ~AclSession();
            

            /// @brief 初始化函数
            /// @return 是否成功
            virtual bool Init();
            

            /// @brief Session 推理会话执行入口 
            /// @return 单次推理是否执行成功
            virtual bool Execute();


            /// @brief 获取当前单次推理过程中 h2d, inf, d2h 三阶段耗时
            /// @return TimeInfoSessionCore 结构体
            TimeInfoSessionCore GetTimeInfo();
    };
}