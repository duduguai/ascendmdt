#define ENABLE_DVPP_INTERFACE
#include <acl/ops/acl_dvpp.h>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cstring>
#include <iomanip>
#include <stdexcept>

#include <acl/acl.h>
#include <aclnnop/aclnn_ops_cv.h>
#include "acl_session.h"

namespace ZWXT::AI{
    
    namespace{
        // 维护模型输入输出信息结构体
        // buffer -> 保存数据的首地址指针
        // size -> 保存数据的大小，单位 Byte
        // dataBuffer -> ACL 用于维护数据块信息的结构体
        struct datainfo{
            void* buffer;
            std::size_t size;
            aclDataBuffer* dataBuffer;
        };

        // Dvpp 内存对齐宽、高计算工具
        // value -> 需要对齐的值
        // factor -> 对齐因子
        uint32_t AlignUp(uint32_t value, uint32_t factor)
        {
            return (value + factor - 1) / factor * factor;
        }

        // 判断发送给 NPU 的指令是否成功下发
        // ret -> ACL 操作返回的判别码
        // OperType -> 操作的名称
        // coutFlg -> 当错误时是否将错误信息打印到终端
        bool check(aclError ret, const std::string& OperType="", bool coutFlg=false)
        {
            if (ret != ACL_SUCCESS)
            {
                if (coutFlg)
                {
                    const char* errMsg = aclGetRecentErrMsg();
                    std::cerr<<std::endl<<"[ERROR] "<<OperType<<" Failed! ACL Error Code: "<<std::to_string(ret)<<std::endl
                             <<"        Recent Msg: "<<std::string(errMsg)<<std::endl;
                }
                return false;
            }
            return true;
        }
    }
    
    class AclSessionBase::Instance
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

            // Aipp 标志位
            bool isAipp{false};

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
            bool isRecordTime{true};

            // 维护会话推理执行参数信息
            SessionExecParam* paramPtr;

            // Base Init: 初始化 ACL 运行环境, 并完成 Model Load
            bool BaseInit()
            {   
                // 创建模型会话推理所需的 Context 和 Stream
                ret = aclrtCreateContext(&context, deviceId);
                if (!check(ret, "ACL Create Context", true)) return false;

                ret = aclrtCreateStream(&stream);
                if (!check(ret, "ACL Create Stream", true)) return false;


                // 查询模型所需内存，创建内存空间，加载模型，创建并加载模型描述
                ret = aclmdlQuerySize(modelPath.c_str(), &modelMemSize, &modelWeightSize);
                if (!check(ret, "Query Model Size", true)) return false;

                ret = aclrtMalloc(&modelMemBuffer, modelMemSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (!check(ret, "Create Mem Space for Model Mem", true)) return false;

                ret = aclrtMalloc(&modelWeightBuffer, modelWeightSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (!check(ret, "Create Mem Space for Model Weights", true)) return false;

                ret = aclmdlLoadFromFileWithMem(modelPath.c_str(), &modelId, modelMemBuffer, modelMemSize, modelWeightBuffer, modelWeightSize);
                if (!check(ret, "Load Model", true)) return false;

                modelDesc = aclmdlCreateDesc();
                ret = aclmdlGetDesc(modelDesc, modelId);
                if (!check(ret, "Create Model Desc", true)) return false;

                return true;
            }

            // 为模型在 NPU 上创建输入输出数据的内存空间
            // ioType -> 为 "input" or "output" 创建空间, 取值空间为 ["input", "output"]
            // modelDesc -> 加载模型时创建的维护模型信息结构体
            // datainfos -> 维护创建空间信息的结构体
            // dataset -> ACL 用于维护模型输入输出的结构体
            // isAipp -> 是否采用 AIPP
            // isDvpp -> 是否采用 DVPP
            // isNmsop -> 是否采用 NMS 算子
            bool prepareDatesetFunc(const std::string& ioType, aclmdlDesc* modelDesc, std::vector<datainfo>& datainfos,
                aclmdlDataset*& dataset, bool isAipp=false, bool isDvpp=false, bool isNmsop=false)
            {
                int ioNum;
                // 创建用于获取模型输入输出数据大小的函数指针
                size_t (*acl_mdl_get_size_by_index)(aclmdlDesc *, size_t);

                // 得到输入、输出的 Tensor 数量以及将函数指针绑定到具体取值函数上
                if (ioType == "input")
                {
                    ioNum = aclmdlGetNumInputs(modelDesc);
                    acl_mdl_get_size_by_index = aclmdlGetInputSizeByIndex;
                }
                else if (ioType == "output")
                {
                    ioNum = aclmdlGetNumOutputs(modelDesc);
                    acl_mdl_get_size_by_index = aclmdlGetOutputSizeByIndex;
                }
                else
                {
                    std::cerr<<"Value Erro: ioType 请从 \"input\" 或 \"output\" 其中选择之一作为输入参数."<<std::endl;
                    return false;
                }

                aclError ret;
                // 为模型的输入输出数据创建 Ascend 提供的专属的 Dateset 类进行管理
                dataset = aclmdlCreateDataset();
                for (int i=0; i<ioNum; ++i)
                {
                    size_t bufferSize = acl_mdl_get_size_by_index(modelDesc, i);
                    void* buffer;
                    if (ioType == "input")
                    {
                        if (! isDvpp)
                        {
                            if (runMode==ACL_HOST)
                            {
                                ret = aclrtMalloc(&buffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
                                if (!check(ret, "Malloc for Input", true)) return false;
                            }
                            else
                            {
                                if (!isAipp) buffer = (*paramPtr).baseParam.dataInBase->at(i).data();
                                else buffer = (*paramPtr).aippParam.dataInAipp->at(i).data();
                            }
                        }
                        else
                        {
                            ret = acldvppMalloc(&buffer, bufferSize);
                            if (!check(ret, "Malloc for Input via Dvpp", true)) return false;
                        }
                    }
                    else
                    {
                        if (runMode==ACL_HOST)
                        {
                            ret = aclrtMalloc(&buffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
                            if (!check(ret, "Malloc for Output", true)) return false;
                        }
                        else
                        {
                            if (!isNmsop) buffer = (*paramPtr).baseParam.dataOutBase->at(i).data();
                            else
                            {
                                if (i==0) buffer = (*paramPtr).nmsopParam.boxes->data();
                                else buffer = (*paramPtr).nmsopParam.scores->data();
                            }
                        }
                    }
                    // 创建管理 NPU 上内存的 DataBuffer 类，并将其添加进 Dataset 中;
                    aclDataBuffer* dataBuffer = aclCreateDataBuffer(buffer, bufferSize);
                    ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
                    if (!check(ret, "Add DataBuffer in Dataset for "+ioType, true)) return false;

                    datainfo dataInfoTemp;
                    dataInfoTemp.buffer = buffer;
                    dataInfoTemp.size = bufferSize;
                    dataInfoTemp.dataBuffer = dataBuffer;

                    datainfos.push_back(std::move(dataInfoTemp));
                }
                return true;
            }

            // 数据传输函数
            template <typename T>
            bool Host2DeviceEP(std::vector<std::vector<T>>& dataInput)
            {
                if (inputDataInfos.size() != dataInput.size())
                {
                    std::cerr<<std::endl<<"[ERROR] 模型输入需要 "<<inputDataInfos.size()<<" Tensor 而传入数据的 Tensor 数量为 "<<dataInput.size()<<"!"<<std::endl;
                    return false;
                }
                for (int i=0; i<inputDataInfos.size(); ++i)
                {
                    ret = aclrtMemcpy(inputDataInfos[i].buffer, inputDataInfos[i].size, dataInput[i].data(), inputDataInfos[i].size, ACL_MEMCPY_HOST_TO_DEVICE);
                    if (!check(ret, "Host Input Transfer to Device", true)) return false;
                }
                return true;
            }
            virtual bool Device2HostEP()
            {
                if (outputDataInfos.size() != (*paramPtr).baseParam.dataOutBase->size())
                {
                    std::cerr<<std::endl<<"[ERROR] 模型输出要求 "<<outputDataInfos.size()<<" Tensor 而 Host 端预设数据的 Tensor 数量为 "<<
                    (*paramPtr).baseParam.dataOutBase->size()<<"!"<<std::endl;
                    return false;
                }

                for (int i=0; i<outputDataInfos.size(); ++i)
                {
                    ret = aclrtMemcpy((*paramPtr).baseParam.dataOutBase->at(i).data(), outputDataInfos[i].size, outputDataInfos[i].buffer, 
                    outputDataInfos[i].size, ACL_MEMCPY_DEVICE_TO_HOST);
                    if (!check(ret, "Device Output Transfer to Host", true)) return false;
                }
                return true;
            }

            // 会话前处理 Base 模块
            bool PreProcessBase()
            {
                if (runMode == ACL_HOST)
                {
                    std::chrono::steady_clock::time_point timeH2D0, timeH2D1;
                    if (!isAipp) 
                    {
                        if (isRecordTime) timeH2D0 = std::chrono::steady_clock::now();
                        if (!Host2DeviceEP(*(paramPtr)->baseParam.dataInBase)) return false;
                        if (isRecordTime)
                        {
                            timeH2D1 = std::chrono::steady_clock::now();
                            ((*paramPtr).timeInfo)->h2d += std::chrono::duration<float, std::milli>(timeH2D1-timeH2D0).count();
                        }
                    }
                    else
                    {
                        if (isRecordTime) timeH2D0 = std::chrono::steady_clock::now();
                        if (!Host2DeviceEP(*(paramPtr)->aippParam.dataInAipp)) return false;
                        if (isRecordTime)
                        {
                            timeH2D1 = std::chrono::steady_clock::now();
                            ((*paramPtr).timeInfo)->h2d += std::chrono::duration<float, std::milli>(timeH2D1-timeH2D0).count();
                        }
                    }
                }
                return true;
            }

            // 会话推理模块
            bool Inference()
            {
                std::chrono::steady_clock::time_point timeInfer0, timeInfer1;
                if (isRecordTime) timeInfer0 = std::chrono::steady_clock::now();
                ret = aclmdlExecute(modelId, inputDataset, outputDataset);
                if (!check(ret, "Model Inferences", true)) return false;
                if (isRecordTime)
                {
                    timeInfer1 = std::chrono::steady_clock::now();
                    ((*paramPtr).timeInfo)->inf += std::chrono::duration<float, std::milli>(timeInfer1-timeInfer0).count();
                }
                return true;
            }

            // 会话后处理 Base 模块
            bool PostProcessBase()
            {
                if (runMode == ACL_HOST)
                {
                    std::chrono::steady_clock::time_point timeD2H0, timeD2H1;
                    if (isRecordTime) timeD2H0 = std::chrono::steady_clock::now();
                    if (!Device2HostEP()) return false;
                    if (isRecordTime)
                    {
                        timeD2H1 = std::chrono::steady_clock::now();
                        ((*paramPtr).timeInfo)->d2h += std::chrono::duration<float, std::milli>(timeD2H1-timeD2H0).count();
                    } 
                }
                return true;
            }

            virtual bool PreProcess()
            {
                if (! PreProcessBase()) return false;
                return true;
            }

            virtual bool PostProcess()
            {
                if (! PostProcessBase()) return false;
                return true;
            }



        public:
            Instance(const std::string& modelpath, int deviceid, SessionExecParam* paramptr, int batchSizeIn, 
                bool needRecordTime, bool isaipp): modelPath(modelpath), deviceId(deviceid), paramPtr(paramptr), 
                batchSize(batchSizeIn), isRecordTime(needRecordTime), isAipp(isaipp)
            {
                if (_count == 0)
                {
                    ret = aclInit(nullptr);
                    if (!check(ret, "ACL Initialize", true)) {throw std::runtime_error("[ERROR] ACL 运行环境初始化失败!");}

                    ret = aclrtGetRunMode(&runMode);
                    if (!check(ret, "ACL Get RunMode", true)) {throw std::runtime_error("[ERROR] ACL 运行模式查询失败!");};

                    std::cout<<std::endl<<"[INFO] ACL Run Mode [ACL_DEVICE:0, ACL_HOST:1]. ";

                    if (runMode==ACL_HOST) std::cout<<"当前设备的模式为 ACL_HOST, 即 EP 模式."<<std::endl;
                    else if(runMode==ACL_DEVICE) std::cout<<"当前设备的模式为 ACL_DEVICE, 即 RC 模式."<<std::endl;
                }
                _count += 1;
            }
            virtual ~Instance()
            {
                // 销毁为模型的输入、输出创建的内存空间以及描述
                for (int i=0; i<inputDataInfos.size(); ++i)
                {
                    if (inputDataInfos[i].buffer) aclrtFree(inputDataInfos[i].buffer);
                    aclDestroyDataBuffer(inputDataInfos[i].dataBuffer);
                }
                for (int i=0; i<outputDataInfos.size(); ++i)
                {
                    if (outputDataInfos[i].buffer) aclrtFree(outputDataInfos[i].buffer);
                    aclDestroyDataBuffer(outputDataInfos[i].dataBuffer);
                }
                if (inputDataset) aclmdlDestroyDataset(inputDataset);
                if (outputDataset) aclmdlDestroyDataset(outputDataset);

                // 销毁模型描述以及卸载模型并清除加载模型权重的内存
                if (modelDesc) aclmdlDestroyDesc(modelDesc);
                if (modelId!=0) aclmdlUnload(modelId);
                if (modelMemBuffer) aclrtFree(modelMemBuffer);
                if (modelWeightBuffer) aclrtFree(modelWeightBuffer);

                // 对 ACL 的运行环境去初始化
                if (stream != nullptr) aclrtDestroyStream(stream);
                if (context != nullptr) aclrtDestroyContext(context);
                if (_count == 1)
                {
                    aclFinalize();
                }
                
                _count -= 1;
            }

            // Instance 实例初始化函数
            virtual bool Init()
            {
                if (!BaseInit()) return false;
                
                // 为模型输入、输出创建描述以及内存空间
                if (!prepareDatesetFunc("input", modelDesc, inputDataInfos, inputDataset, isAipp)) return false;
                if (!prepareDatesetFunc("output", modelDesc, outputDataInfos, outputDataset, isAipp)) return false;
                return true;
            }

            virtual bool Execute()
            {
                ret = aclrtSetCurrentContext(context);
                if (!check(ret, "Set Current Context For Model", true)) return false;

                if (!PreProcess()) return false;
                if (!Inference()) return false;
                if (!PostProcess()) return false;
                return true;
            }
    };

    AclSessionBase::AclSessionBase(const std::string& modelPath, int deviceId, SessionExecParam* paramptr, int batchSizeIn, 
                bool needRecordTime, bool isaipp)
                {
                    inst_ = std::make_unique<Instance>(modelPath, deviceId, paramptr, batchSizeIn, needRecordTime, isaipp);
                }
    AclSessionBase::~AclSessionBase(){};

    bool AclSessionBase::Init(){if (! inst_ -> Init()) return false; return true;}
    bool AclSessionBase::Execute(){return inst_ -> Execute();}



    namespace CV{
        class AclSessionYOLO::Instance : public AclSessionBase::Instance
        {
            protected:
                bool isDvpp{false};
                bool isNmsop{false};

            // 使用 DVPP 需要引入的变量
                // LetterBox 填充像素值
                double lettPad{114};

                // 模型输入图像尺寸信息
                int imWidth{640};
                int imHeight{640};

                // DVPP 通道描述符
                acldvppChannelDesc* JPEGD{nullptr};
                acldvppChannelDesc* VPC{nullptr};

                // 预设 binary 图像默认大小 (Bytes)
                int defaultBinSz{2*1024*1024};
                std::unique_ptr<std::vector<void*>> imsBinVecPtr{nullptr};
                // 预设经 DVPP 解码后图像默认大小 (Bytes)
                int defaultDecSz{4*1024*1024};
                std::unique_ptr<std::vector<void*>> imsDecVecPtr{nullptr};

                // 使用 DVPP 进行 LetterBox 所需操作描述符
                acldvppBatchPicDesc* srcBtPicDesc{nullptr};
                acldvppBatchPicDesc* dstBtPicDesc{nullptr};
                std::unique_ptr<std::vector<uint32_t>> roiNumsPtr{nullptr};
                std::unique_ptr<std::vector<acldvppRoiConfig*>> cropAreasPtr{nullptr};
                std::unique_ptr<std::vector<acldvppBorderConfig*>> borderCfgsPtr{nullptr};
                acldvppResizeConfig* resizeCfg{nullptr};

                // 创建 DVPP 通道函数
                // desc -> 通道描述符
                // mode -> 通道设定模式
                // modename -> 通道设定模式的名称
                bool CreateDvppChannel(acldvppChannelDesc*& desc, acldvppChannelMode mode, std::string modename)
                {
                    desc = acldvppCreateChannelDesc();
                    ret = acldvppSetChannelDescMode(desc, mode);
                    if (!check(ret, "Set ChannelDesc Mode: "+modename, true)) return false;
                    ret = acldvppCreateChannel(desc);
                    if (!check(ret, "Create Channel: "+modename, true)) return false;

                    return true;
                }
            
            // 使用 NMS OP 需要引入的变量
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

            public:
                Instance(const std::string& modelPath, int deviceId, SessionExecParam* paramptr, int batchSizeIn, bool needRecordTime, 
                    bool isaipp, bool isdvpp, bool isnmsop, std::optional<int> imwidth, std::optional<int> imheight, std::optional<double> lettpad):
                AclSessionBase::Instance(modelPath, deviceId, paramptr, batchSizeIn, needRecordTime, isaipp), isDvpp(isdvpp), isNmsop(isnmsop)
                {
                    if (imwidth) imWidth = *imwidth;
                    if (imheight) imHeight = *imheight;
                    if (lettpad) lettPad = *lettpad;
                }
                ~Instance()
                {
                    if (isDvpp)
                    {
                        // 销毁 LetterBox 中的 Roiconfig 和 BorderConfig 以及 ResizeConfig
                        if (resizeCfg) acldvppDestroyResizeConfig(resizeCfg);
                        if (cropAreasPtr)
                        {
                            for (int i=0; i<batchSize; ++i) {if (cropAreasPtr->at(i)) acldvppDestroyRoiConfig(cropAreasPtr->at(i));}
                        }
                        if (borderCfgsPtr)
                        {
                            for (int i=0; i<batchSize; ++i) {if (borderCfgsPtr->at(i)) acldvppDestroyBorderConfig(borderCfgsPtr->at(i));}
                        }
                        // 销毁 LetterBox 输入和输出图像 Batch 描述符
                        if (srcBtPicDesc) acldvppDestroyBatchPicDesc(srcBtPicDesc);
                        if (dstBtPicDesc) acldvppDestroyBatchPicDesc(dstBtPicDesc);

                        // 销毁为 Binary 数据和 Decode 数据创建的内存空间
                        if (imsDecVecPtr)
                        {
                            for (int i=0; i<batchSize; ++i){if (imsDecVecPtr->at(i)) acldvppFree(imsDecVecPtr->at(i));}
                        }
                        if (imsBinVecPtr)
                        {
                            for (int i=0; i<batchSize; ++i){if (imsBinVecPtr->at(i)) aclrtFree(imsBinVecPtr->at(i));}
                        }

                        // 销毁通道描述符
                        if (VPC) {acldvppDestroyChannel(VPC); acldvppDestroyChannelDesc(VPC);}
                        if (JPEGD) {acldvppDestroyChannel(JPEGD); acldvppDestroyChannelDesc(JPEGD);}
                    }

                    if (isNmsop)
                    {
                        // 释放 WorkSpace 内存空间
                        if (workSpaceAddr) aclrtFree(workSpaceAddr);

                        // 销毁 IntArray 与 FloatArray
                        if (maxDetsPerClaIntArr) aclDestroyIntArray(maxDetsPerClaIntArr);
                        if (confThFloArr) aclDestroyFloatArray(confThFloArr);
                        if (iouThFloArr) aclDestroyFloatArray(iouThFloArr);

                        // 销毁 Tensor
                        if (boxes) aclDestroyTensor(boxes);
                        if (scores) aclDestroyTensor(scores);
                        if (indices) aclDestroyTensor(indices);

                        // EP模式，销毁为 Indices 在 Device 上的内存空间
                        if (runMode == ACL_HOST)
                        {
                            if (indicesAddr) aclrtFree(indicesAddr);
                        }
                    }
                }

                bool Init() override
                {
                    if (!BaseInit()) return false;

                    if (!prepareDatesetFunc("input", modelDesc, inputDataInfos, inputDataset, isAipp, isDvpp, isNmsop)) return false;
                    if (!prepareDatesetFunc("output", modelDesc, outputDataInfos, outputDataset, isAipp, isDvpp, isNmsop)) return false;

                    // 使用 DVPP 时需要额外针对 DVPP 需要使用的资源进行初始化
                    if (isDvpp)
                    {
                        // 计算模型输入图像宽、高经过对齐后的值 
                        int imWidthStride = AlignUp(imWidth, 64); // * 64 是针对于 Atlas 推理系列产品 (310B / 310P), 后续需要进一步更新泛用性
                        int imHeightStride = AlignUp(imHeight, 16); // * 16 是针对于 Atlas 推理系列产品 (310B / 310P), 后续需要进一步更新泛用性
                        int bytesPerIm = imWidth*imHeight*3/2; // * 单位为 Bytes, 此处计算的是经 DVPP Decode 和 LetterBox 后单张图像的大小，格式采用的是 YUV_420_SP, 后续需要进一步更新泛用性

                        // 创建 DVPP 操作所需要的 Channel 
                        if (!CreateDvppChannel(JPEGD, DVPP_CHNMODE_JPEGD, "JPEGD")) return false;
                        if (!CreateDvppChannel(VPC, DVPP_CHNMODE_VPC, "VPC")) return false;

                        // 创建保存原始 Binary 数据的地址变量
                        imsBinVecPtr = std::make_unique<std::vector<void*>>(batchSize);
                        for (int i=0; i<batchSize; ++i)
                        {
                            // EP 模式下，需要在 Device 端创建保存 Binary 数据的内存空间
                            if (runMode==ACL_HOST)
                            {
                                ret = aclrtMalloc(&(imsBinVecPtr->at(i)), defaultBinSz, ACL_MEM_MALLOC_HUGE_FIRST);
                                if (!check(ret, "使用 DVPP, 在 Device 上为原始二进制数据创建内存空间", true)) return false;
                            }
                            // RC 模式下，仅需要将 param 中 DVPP 下输入参数的首地址绑定到 vector 中
                            else
                            {
                                imsBinVecPtr->at(i) = (*paramPtr).dvppParam.dataInDvpp->at(i).data();
                            }
                        }

                        // 为 DVPP 解码后的中间数据创建内存空间
                        imsDecVecPtr = std::make_unique<std::vector<void*>>(batchSize);
                        for (int i=0; i<batchSize; ++i)
                        {
                            ret = acldvppMalloc(&(imsDecVecPtr->at(i)), defaultDecSz);
                            if (!check(ret, "使用 DVPP, 为 Decode 中间数据创建内存空间", true)) return false;
                        }

                        // 为 DVPP 实现 LetterBox 构造所需的 输入(src) 和 输出(dst) 信息描述符
                        srcBtPicDesc = acldvppCreateBatchPicDesc(batchSize);
                        dstBtPicDesc = acldvppCreateBatchPicDesc(batchSize);
                        acldvppPicDesc* dstPicDesc;
                        void* dstBufferPtr = inputDataInfos[0].buffer; // 当前 DVPP 仅支持模型输入中当且仅当第一个参数为输入图像数据的情况，后续可能考虑进一步的泛用性更新
                        for (int i=0; i<batchSize; ++i)
                        {
                            dstPicDesc = acldvppGetPicDesc(dstBtPicDesc, i);
                            acldvppSetPicDescData(dstPicDesc, dstBufferPtr);
                            acldvppSetPicDescSize(dstPicDesc, bytesPerIm);
                            acldvppSetPicDescFormat(dstPicDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
                            acldvppSetPicDescWidth(dstPicDesc, imWidth);
                            acldvppSetPicDescHeight(dstPicDesc, imHeight);
                            acldvppSetPicDescWidthStride(dstPicDesc, imWidthStride);
                            acldvppSetPicDescHeightStride(dstPicDesc, imHeightStride);

                            char* dstCharPtr = (char*)dstBufferPtr + bytesPerIm;
                            dstBufferPtr = dstCharPtr;
                        }
                        // ret = acldvppDestroyPicDesc(dstPicDesc);
                        // if (!check(ret, "销毁 DVPP 暂存图像信息的中间变量", true)) return false;

                        // 为 DVPP 实现 LetterBox 构造所需的操作参数描述
                        roiNumsPtr = std::make_unique<std::vector<uint32_t>>(batchSize, 1);
                        cropAreasPtr = std::make_unique<std::vector<acldvppRoiConfig*>>(batchSize);
                        borderCfgsPtr = std::make_unique<std::vector<acldvppBorderConfig*>>(batchSize);
                        for (int i=0; i<batchSize; ++i)
                        {
                            cropAreasPtr->at(i) = acldvppCreateRoiConfig(0, 0, 0, 0);
                            borderCfgsPtr->at(i) = acldvppCreateBorderConfig();
                            acldvppSetBorderConfigBorderType(borderCfgsPtr->at(i), BORDER_CONSTANT);
                            acldvppSetBorderConfigValue(borderCfgsPtr->at(i), 0, lettPad);
                            acldvppSetBorderConfigValue(borderCfgsPtr->at(i), 1, 128);
                            acldvppSetBorderConfigValue(borderCfgsPtr->at(i), 2, 128);
                        }
                        resizeCfg = acldvppCreateResizeConfig();
                    }
                    return true;
                }

                // 使用 NMS 算子需要的独立的初始化函数
                // clsNum -> 模型推理输出的类别数量
                // maxDetsPerCla -> 每个类别经 NMS 后最大保留的预测数量
                // confTh -> 置信度阈值
                // iouTh -> 交并比阈值
                bool InitForNmsOp(int clsNum, int anchorsNum, int64_t maxDetsPerCla, float confTh, float iouTh)
                {
                    // 确定输出 Tensor 的形状
                    std::vector<int64_t> boxShape{static_cast<int64_t>(batchSize), static_cast<int64_t>(anchorsNum), 4};
                    std::vector<int64_t> boxStride{static_cast<int64_t>(4*anchorsNum), 4, 1};
                    std::vector<int64_t> scoreShape{static_cast<int64_t>(batchSize), static_cast<int64_t>(clsNum), static_cast<int64_t>(anchorsNum)};
                    std::vector<int64_t> scoreStride{static_cast<int64_t>(anchorsNum*clsNum), static_cast<int64_t>(anchorsNum), 1};
                    std::vector<int64_t> indicesShape{static_cast<int64_t>(batchSize*maxDetsPerCla*clsNum), 3};
                    std::vector<int64_t> indicesStride{3, 1};

                    // 对于 EP 模式，需要在 Device 上为 NMS OP 输出 Indices 创建内存空间，对于 RC 模式，直接将 param 中的输出与其绑定
                    if (runMode == ACL_HOST)
                    {
                        indicesDevSz = batchSize*maxDetsPerCla*clsNum*3*sizeof(int32_t);
                        ret = aclrtMalloc(&indicesAddr, indicesDevSz, ACL_MEM_MALLOC_HUGE_FIRST);
                        if (!check(ret, "NMS OP, 为算子输出的筛选 Indices 在 Device 上创建内存空间", true)) return false;
                    }
                    else
                    {
                        indicesAddr = (*paramPtr).nmsopParam.indices->data();
                    }


                    // 构建 NMS OP 需要的输入和输出 Tensor 
                    boxes = aclCreateTensor(boxShape.data(), boxShape.size(), aclDataType::ACL_FLOAT, boxStride.data(), 0, aclFormat::ACL_FORMAT_ND, boxShape.data(), boxShape.size(), outputDataInfos[0].buffer);
                    scores = aclCreateTensor(scoreShape.data(), scoreShape.size(), aclDataType::ACL_FLOAT, scoreStride.data(), 0, aclFormat::ACL_FORMAT_ND, scoreShape.data(), scoreShape.size(), outputDataInfos[1].buffer);
                    indices = aclCreateTensor(indicesShape.data(), indicesShape.size(), aclDataType::ACL_INT32, indicesStride.data(), 0, aclFormat::ACL_FORMAT_ND, indicesShape.data(), indicesShape.size(), indicesAddr);

                    // 构建 NMS OP 执行所需的 Array 常量
                    maxDetsPerClaIntArr = aclCreateIntArray(&maxDetsPerCla, 1);
                    confThFloArr = aclCreateFloatArray(&confTh, 1);
                    iouThFloArr = aclCreateFloatArray(&iouTh, 1);

                    // 计算 workSpaceSize 并完成 workSpace 内存空间分配
                    aclnnStatus status;
                    status = aclnnNonMaxSuppressionGetWorkspaceSize(boxes, scores, maxDetsPerClaIntArr, iouThFloArr, confThFloArr, 1, indices, &workSpaceSz, &executor);
                    if (status != 0) 
                    {
                        std::cout<<"[ERROR] NMS OP, 确定 NMS OP 执行所需的 Work Space Size Failed! Error Code: "<<status<<std::endl;
                        if (status == 361001) std::cout<<"[ERROR] 当前设备型号不支持该算子执行！"<<std::endl;
                        return false;
                    }
                    ret = aclrtMalloc(&workSpaceAddr, workSpaceSz, ACL_MEM_MALLOC_HUGE_FIRST);
                    if (!check(ret, "NMS OP, 为 NMS OP 算子执行创建工作所需的内存空间", true)) return false;

                    return true;
                }
                
                // DVPP 解码 JPEG Binary 数据函数
                bool DvppDecode()
                {
                    int oriSz, decSz;
                    int32_t components;
                    uint32_t width, height, widthStride, heightStride;
                    acldvppPicDesc* imDecPicDesc;
                    std::chrono::steady_clock::time_point timeH2d0, timeH2d1, timeDec0, timeDec1;
                    
                    for (int i=0; i<batchSize; ++i)
                    {
                        oriSz = (*paramPtr).dvppParam.dataInDvpp->at(i).size();
                        // EP 模式下首先需要将 Host 内存中的数据转移到 Device 内存上
                        if (runMode==ACL_HOST)
                        {
                            if (isRecordTime) timeH2d0 = std::chrono::steady_clock::now();
                            if (oriSz > defaultBinSz)
                            {
                                std::cout<<"[INFO] 输入图像原始二进制数据大小大于默认 2 MB, 需要在 Device 上重新分配内存空间."<<std::endl;
                                ret = aclrtFree(imsBinVecPtr->at(i)); if (!check(ret, "释放为二进制数据默认申请的内存空间", true)) return false;
                                ret = aclrtMalloc(&(imsBinVecPtr->at(i)), oriSz, ACL_MEM_MALLOC_HUGE_FIRST);
                                if (!check(ret, "为二进制数据重新申请内存空间", true)) return false;
                            }
                            ret = aclrtMemcpy(imsBinVecPtr->at(i), oriSz, (*paramPtr).dvppParam.dataInDvpp->at(i).data(), oriSz, ACL_MEMCPY_HOST_TO_DEVICE);
                            if (!check(ret, "DVPP, Host Data Transfer To Device", true)) return false;
                            if (isRecordTime)
                            {
                                timeH2d1 = std::chrono::steady_clock::now();
                                (*paramPtr).timeInfo->h2d += std::chrono::duration<float, std::milli>(timeH2d1-timeH2d0).count();
                            }
                        }
                        // RC 模式则直接使用 param.dvppParam.dataInDvpp 中保存的数据

                    // DVPP Decode 信息加载
                        if (isRecordTime) timeDec0 = std::chrono::steady_clock::now();
                        // 获取图像数据的基本信息 (在 CPU 上处理的，不需要使用 Device 上的内存)
                        ret = acldvppJpegGetImageInfo((*paramPtr).dvppParam.dataInDvpp->at(i).data(), oriSz, &width, &height, &components);
                        if (!check(ret, "DVPP, 根据原始二进制数据获取图像基本信息", true)) return false;
                        (*paramPtr).dvppParam.letterboxMeta->at(i).imOriW = width;
                        (*paramPtr).dvppParam.letterboxMeta->at(i).imOriH = height;
                        widthStride = AlignUp(width, 64); heightStride = AlignUp(height, 16); // * 此处针对 Atlas 推理产品对 宽stride*高stride 有 64*16 对齐的约束
                        
                        // 判断是否需要重新为 Decode 图像分配内存空间
                        decSz = widthStride*heightStride*3/2; // * 此处计算大小仅针对 YUV_420_SP 格式，后续考虑泛用性更新
                        if (decSz > defaultDecSz)
                        {
                            std::cout<<"[INFO] 解码后图像数据所需空间大于默认 4 MB, 需要在 Device 上重新分配内存空间."<<std::endl;
                            ret = acldvppFree(imsDecVecPtr->at(i)); if (!check(ret, "释放为解码数据默认申请的内存空间", true)) return false;
                            ret = acldvppMalloc(&(imsDecVecPtr->at(i)), decSz);
                            if (!check(ret, "为解码数据重新申请内存空间", true)) return false;
                        }
                        
                        // 为 Decode 图像填充描述符
                        imDecPicDesc = acldvppGetPicDesc(srcBtPicDesc, i);
                        acldvppSetPicDescData(imDecPicDesc, imsDecVecPtr->at(i));
                        acldvppSetPicDescSize(imDecPicDesc, decSz);
                        acldvppSetPicDescFormat(imDecPicDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
                        acldvppSetPicDescWidth(imDecPicDesc, width);
                        acldvppSetPicDescHeight(imDecPicDesc, height);
                        acldvppSetPicDescWidthStride(imDecPicDesc, widthStride);
                        acldvppSetPicDescHeightStride(imDecPicDesc, heightStride);

                        // 挂起 DVPP 解码任务
                        ret = acldvppJpegDecodeAsync(JPEGD, imsBinVecPtr->at(i), oriSz, imDecPicDesc, stream);
                        if (!check(ret, "DVPP, JPEG 解码任务挂载", true)) return false;

                        if (isRecordTime)
                        {
                            timeDec1 = std::chrono::steady_clock::now();
                            (*paramPtr).timeInfo->ld += std::chrono::duration<float, std::milli>(timeDec1-timeDec0).count();
                        }
                    }

                    if (isRecordTime) timeDec0 = std::chrono::steady_clock::now();
                    ret = aclrtSynchronizeStream(stream);
                    if (!check(ret, "DVPP, 等待 JPEG 任务解码同步", true)) return false;
                    if (isRecordTime)
                    {
                        timeDec1 = std::chrono::steady_clock::now();
                        (*paramPtr).timeInfo->ld += std::chrono::duration<float, std::milli>(timeDec1-timeDec0).count();
                    }

                    return true;
                }

                // 通过 DVPP 实现 LetterBox
                bool LetterBox()
                {
                    acldvppPicDesc* srcPicDesc;
                    acldvppPicDesc* dstPicDesc;
                    float scaleW, scaleH, scaleFactor;
                    uint32_t width, height, rzwidth, rzheight, padL, padT, padR, padB;
                    std::chrono::steady_clock::time_point timeLett0, timeLett1;

                    if (isRecordTime) timeLett0 = std::chrono::steady_clock::now();
                    for (int i=0; i<batchSize; ++i)
                    {
                        srcPicDesc = acldvppGetPicDesc(srcBtPicDesc, i);
                        width = acldvppGetPicDescWidth(srcPicDesc);
                        height = acldvppGetPicDescHeight(srcPicDesc);

                        // 计算 LetterBox 变换参数
                        scaleW = static_cast<float>(imWidth) / static_cast<float>(width);
                        scaleH = static_cast<float>(imHeight) / static_cast<float>(height);
                        scaleFactor = std::min(scaleW, scaleH);
                        rzwidth = width * scaleFactor; rzheight = height * scaleFactor;
                        (*paramPtr).dvppParam.letterboxMeta->at(i).scale = scaleFactor;

                        // 注意！ DVPP 中使用 Padding 工具时需要注意 Padding 值需要是偶数，否则在执行过程中会引发内存分配错误
                        padL = (imWidth-rzwidth)/2; padL = (padL%2==0) ? padL:padL+1;
                        padT = (imHeight-rzheight)/2; padT = (padT%2==0) ? padT:padT+1;
                        padR = imWidth - rzwidth - padL; padR = (padR%2==0) ? padR:padR+1;
                        padB = imHeight - rzheight - padT; padB = (padB%2==0) ? padB:padB+1;

                        (*paramPtr).dvppParam.letterboxMeta->at(i).left = padL;
                        (*paramPtr).dvppParam.letterboxMeta->at(i).top = padT;
                        (*paramPtr).dvppParam.letterboxMeta->at(i).right = padR;
                        (*paramPtr).dvppParam.letterboxMeta->at(i).bottom = padB;

                        // 为 cropAreas 赋值，指定 srcPic 中有效的图像区域
                        acldvppSetRoiConfig(cropAreasPtr->at(i), 0, width-1, 0, height-1);
                        // 为 borderCfgs 赋值，指定 Pad 的区域
                        acldvppSetBorderConfigLeft(borderCfgsPtr->at(i), padL);
                        acldvppSetBorderConfigTop(borderCfgsPtr->at(i), padT);
                        acldvppSetBorderConfigRight(borderCfgsPtr->at(i), padR);
                        acldvppSetBorderConfigBottom(borderCfgsPtr->at(i), padB);
                    }

                    // DVPP，挂载 Crop + Resize + Padding 任务
                    ret = acldvppVpcBatchCropResizeMakeBorderAsync(VPC, srcBtPicDesc, roiNumsPtr->data(), batchSize, dstBtPicDesc, 
                            cropAreasPtr->data(), borderCfgsPtr->data(), resizeCfg, stream);
                    if (!check(ret, "DVPP, 挂载 Crop + Resize + Padding 任务", true)) return false;

                    // DVPP, 等待 Crop + Resize + Padding 任务同步
                    ret = aclrtSynchronizeStream(stream);
                    if (!check(ret, "DVPP, 等待 Crop + Resize + Padding 任务同步", true)) return false;

                    if (isRecordTime)
                    {
                        timeLett1 = std::chrono::steady_clock::now();
                        (*paramPtr).timeInfo->let += std::chrono::duration<float, std::milli>(timeLett1-timeLett0).count();
                    }

                    return true;
                }

                bool PreProcess() override
                {
                    if (!isDvpp)
                    {
                        if (!PreProcessBase()) return false;
                    }

                    return true;
                }

                bool NmsOpExecute()
                {
                    // NMS OP 算子执行
                    std::chrono::steady_clock::time_point timeNms0, timeNms1;
                    if (isRecordTime) timeNms0 = std::chrono::steady_clock::now();
                    aclnnStatus status;
                    status = aclnnNonMaxSuppressionGetWorkspaceSize(boxes, scores, maxDetsPerClaIntArr, iouThFloArr, confThFloArr, 1, indices, &workSpaceSz, &executor);
                    if (status != 0) 
                    {
                        std::cout<<"[ERROR] NMS OP, 确定 NMS OP 执行所需的 Work Space Size Failed! Error Code: "<<status<<std::endl;
                        if (status == 361001) std::cout<<"[ERROR] 当前设备型号不支持该算子执行！"<<std::endl;
                        return false;
                    }
                    status = aclnnNonMaxSuppression(workSpaceAddr, workSpaceSz, executor, stream);
                    if (status != 0) 
                    {
                        std::cout<<"[ERROR] NMS OP, 挂载 NMS 算子执行任务 Failed! Error Code: "<<status<<std::endl;
                        if (status == 361001) std::cout<<"[ERROR] 当前设备型号不支持该算子执行！"<<std::endl;
                        return false;
                    }
                    ret = aclrtSynchronizeStream(stream);
                    if (!check(ret, "NMS OP, 等待 NMS 算子执行任务同步", true)) return false;
                    if (isRecordTime)
                    {
                        timeNms1 = std::chrono::steady_clock::now();
                        (*paramPtr).timeInfo->nms += std::chrono::duration<float, std::milli>(timeNms1-timeNms0).count();
                    }
                    return true;
                }

                bool PostProcess() override
                {
                    if (!isNmsop)
                    {
                        if (!PostProcessBase()) return false;
                    }
                    else
                    {
                        if (!NmsOpExecute()) return false;

                        if (runMode == ACL_HOST)
                        {
                            std::chrono::steady_clock::time_point timeD2H0, timeD2H1;
                            if (isRecordTime) timeD2H0 = std::chrono::steady_clock::now();
                            ret = aclrtMemcpyAsync((*paramPtr).nmsopParam.boxes->data(), outputDataInfos[0].size, outputDataInfos[0].buffer, outputDataInfos[0].size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
                            if (!check(ret, "NMS OP, EP 模式下挂载 box 数据 D2H 任务", true)) return false;
                            ret = aclrtMemcpyAsync((*paramPtr).nmsopParam.scores->data(), outputDataInfos[1].size, outputDataInfos[1].buffer, outputDataInfos[1].size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
                            if (!check(ret, "NMS OP, EP 模式下挂载 score 数据 D2H 任务", true)) return false;
                            ret = aclrtMemcpyAsync((*paramPtr).nmsopParam.indices->data(), indicesDevSz, indicesAddr, indicesDevSz, ACL_MEMCPY_DEVICE_TO_HOST, stream);
                            if (!check(ret, "NMS OP, EP 模式下挂载 indices 数据 D2H 任务", true)) return false;
                            ret = aclrtSynchronizeStream(stream);
                            if (!check(ret, "NMS OP, EP 模式下等待模型输出数据 D2H 传输同步", true)) return false;
                            
                            if (isRecordTime)
                            {
                                timeD2H1 = std::chrono::steady_clock::now();
                                (*paramPtr).timeInfo->d2h += std::chrono::duration<float, std::milli>(timeD2H1-timeD2H0).count();
                            }
                        }
                    }
                    return true;
                }


                void UpdateAddrParam(int index)
                {
                    imsBinVecPtr->at(index) = (*paramPtr).dvppParam.dataInDvpp->at(index).data();
                }        
        };

        AclSessionYOLO::AclSessionYOLO(const std::string& modelPath, int deviceId, SessionExecParam* paramptr, int batchSizeIn, 
                bool needRecordTime, bool isaipp, bool isdvpp, bool isnmsop, std::optional<int> imwidth, std::optional<int> imheight, 
                std::optional<double> lettpad)
                {
                    inst_ = std::make_unique<Instance>(modelPath, deviceId, paramptr, batchSizeIn, needRecordTime, isaipp, isdvpp, isnmsop, imwidth, 
                        imheight, lettpad);
                }
        AclSessionYOLO::~AclSessionYOLO(){}
        bool AclSessionYOLO::Init(){return inst_->Init();}
        bool AclSessionYOLO::JpegDecodeViaDvpp(){return inst_->DvppDecode();}
        bool AclSessionYOLO::PreprocessViaDvpp(){return inst_->LetterBox();}
        bool AclSessionYOLO::Execute(){return inst_ -> Execute();}
        void AclSessionYOLO::UpdateAddrParam(int index){inst_->UpdateAddrParam(index);}
        bool AclSessionYOLO::InitForNmsOp(int clsNum, int anchorsNum, int64_t maxDetsPerCla, float confTh, float iouTh)
        {return inst_->InitForNmsOp(clsNum, anchorsNum, maxDetsPerCla, confTh, iouTh);}
    }
}