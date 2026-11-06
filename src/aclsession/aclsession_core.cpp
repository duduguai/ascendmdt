#include <iostream>
#include <chrono>
#include <stdexcept>
#include "aclsession_core.h"

namespace AI
{
    bool AclSession::check(const std::string& OperType, bool coutFlg)
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

    bool AclSession::BaseInit()
    {
        // 创建模型会话推理所需的 Context 和 Stream
        ret = aclrtCreateContext(&context, deviceId);
        if (!check("ACL Create Context", true)) return false;

        ret = aclrtCreateStream(&stream);
        if (!check("ACL Create Stream", true)) return false;


        // 查询模型所需内存，创建内存空间，加载模型，创建并加载模型描述
        ret = aclmdlQuerySize(modelPath.c_str(), &modelMemSize, &modelWeightSize);
        if (!check("Query Model Size", true)) return false;

        ret = aclrtMalloc(&modelMemBuffer, modelMemSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!check("Create Mem Space for Model Mem", true)) return false;

        ret = aclrtMalloc(&modelWeightBuffer, modelWeightSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!check("Create Mem Space for Model Weights", true)) return false;

        ret = aclmdlLoadFromFileWithMem(modelPath.c_str(), &modelId, modelMemBuffer, modelMemSize, modelWeightBuffer, modelWeightSize);
        if (!check("Load Model", true)) return false;

        modelDesc = aclmdlCreateDesc();
        ret = aclmdlGetDesc(modelDesc, modelId);
        if (!check("Create Model Desc", true)) return false;

        // 为模型输入、输出创建描述以及内存空间 {0: input, 1: output}
        if (!PrepareDateset(0, modelDesc, inputDataInfos, inputDataset)) return false;
        if (!PrepareDateset(1, modelDesc, outputDataInfos, outputDataset)) return false;

        return true;
    }

    bool AclSession::CheckTensorConsist(int requireNum, int mode)
    {
        std::cout<<"[ERROR] 派生类中需要对 CheckTensorConsist 进行重载, 基类该函数不提供任何功能.";
        return false;
    }

    bool AclSession::CheckBytesConsist(int bufferSz, int mode, int idx)
    {
        std::cout<<"[ERROR] 派生类中需要对 CheckBytesConsist 进行重载, 基类该函数不提供任何功能.";
        return false;
    }

    bool AclSession::PrepareBuffer(int idx, void*& buffer, int bufferSz, int mode)
    {
        std::cout<<"[ERROR] 派生类中需要对 PrepareBuffer 进行重载, 基类该函数不提供任何功能."<<std::endl;
        return false;
    }

    bool AclSession::PrepareDateset(int ioMode, aclmdlDesc* modelDesc, std::vector<datainfo>& datainfos,
                aclmdlDataset*& dataset)
    {
        int ioNum;
        // 创建用于获取模型输入输出数据大小的函数指针
        size_t (*acl_mdl_get_size_by_index)(aclmdlDesc *, size_t);

        // 得到输入、输出的 Tensor 数量以及将函数指针绑定到具体取值函数上
        if (ioMode == 0)
        {
            ioNum = aclmdlGetNumInputs(modelDesc);
            acl_mdl_get_size_by_index = aclmdlGetInputSizeByIndex;
        }
        else if (ioMode == 1)
        {
            ioNum = aclmdlGetNumOutputs(modelDesc);
            acl_mdl_get_size_by_index = aclmdlGetOutputSizeByIndex;
        }
        else
        {
            std::cerr<<"[Erro] ioMode 赋值错误, 取值空间为 {0: input, 1: output}."<<std::endl;
            return false;
        }

        if (!CheckTensorConsist(ioNum, ioMode)) return false;

        // 为模型的输入输出数据创建 Ascend 提供的专属的 Dateset 类进行管理
        dataset = aclmdlCreateDataset();
        for (int i=0; i<ioNum; ++i)
        {
            size_t bufferSize = acl_mdl_get_size_by_index(modelDesc, i);

            if (!CheckBytesConsist(bufferSize, ioMode, i)) return false;

            void* buffer;

            if (!PrepareBuffer(i, buffer, bufferSize, ioMode)) return false;
            
            // 创建管理 NPU 上内存的 DataBuffer 类，并将其添加进 Dataset 中;
            aclDataBuffer* dataBuffer = aclCreateDataBuffer(buffer, bufferSize);
            ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
            if (!check("Add DataBuffer in Dataset for "+ioType[ioMode], true)) return false;

            datainfo dataInfoTemp;
            dataInfoTemp.buffer = buffer;
            dataInfoTemp.size = bufferSize;
            dataInfoTemp.dataBuffer = dataBuffer;

            datainfos.push_back(std::move(dataInfoTemp));
        }
        return true;
    }

    bool AclSession::Host2DeviceEP()
    {
        std::cout<<"[ERROR] 派生类中需要对 Host2DeviceEP 进行重载, 基类该函数不提供任何功能."<<std::endl;
        return false;
    }

    bool AclSession::Device2HostEP()
    {
        std::cout<<"[ERROR] 派生类中需要对 Device2HostEP 进行重载, 基类该函数不提供任何功能."<<std::endl;
        return false;
    }

    bool AclSession::PreProcess()
    {
        std::chrono::steady_clock::time_point timeH2D0, timeH2D1;
        if (timeInfoPtr)timeH2D0 = std::chrono::steady_clock::now();

        if (! Host2DeviceEP()) return false;

        if (timeInfoPtr)
        {
            timeH2D1 = std::chrono::steady_clock::now();
            timeInfoPtr -> h2d += std::chrono::duration<float, std::milli>(timeH2D1-timeH2D0).count();
        }

        return true;
    }

    bool AclSession::Inference()
    {
        std::chrono::steady_clock::time_point timeInf0, timeInf1;
        if (timeInfoPtr)timeInf0 = std::chrono::steady_clock::now();

        ret = aclmdlExecute(modelId, inputDataset, outputDataset);
        if (!check("Model Inferenceing", true)) return false;

        if (timeInfoPtr)
        {
            timeInf1 = std::chrono::steady_clock::now();
            timeInfoPtr->inf += std::chrono::duration<float, std::milli>(timeInf1-timeInf0).count();
        }
        return true;
    }

    bool AclSession::PostProcess()
    {
        std::chrono::steady_clock::time_point timeD2H0, timeD2H1;
        if (timeInfoPtr)timeD2H0 = std::chrono::steady_clock::now();

        if (!Device2HostEP()) return false;

        if (timeInfoPtr)timeD2H1 = std::chrono::steady_clock::now();
        timeInfoPtr->d2h += std::chrono::duration<float, std::milli>(timeD2H1-timeD2H0).count();

        return true;
    }


    AclSession::AclSession(const std::string& modelpath, int deviceid, int batchsize, bool needrecordtime):
    modelPath(modelpath), deviceId(deviceid), batchSize(batchsize)
    {
        if (_count == 0)
        {
            ret = aclInit(nullptr);
            if (!check("ACL Initialize", true)) {throw std::runtime_error("[ERROR] ACL 运行环境初始化失败!");}

            ret = aclrtGetRunMode(&runMode);
            if (!check("ACL Get RunMode", true)) {throw std::runtime_error("[ERROR] ACL 运行模式查询失败!");}

            std::cout<<std::endl<<"[INFO] ACL Run Mode [ACL_DEVICE:0, ACL_HOST:1]. ";

            if (runMode==ACL_HOST) std::cout<<"当前设备的模式为 ACL_HOST, 即 EP 模式."<<std::endl;
            else if(runMode==ACL_DEVICE) std::cout<<"当前设备的模式为 ACL_DEVICE, 即 RC 模式."<<std::endl;
        }

        if (needrecordtime) timeInfoPtr = std::make_unique<TimeInfoSessionCore>();
        _count += 1;
    }


    AclSession::~AclSession()
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


    bool AclSession::Init()
    {
        return BaseInit();
    }


    bool AclSession::Execute()
    {
        if (timeInfoPtr) timeInfoPtr->Init();

        ret = aclrtSetCurrentContext(context);
        if (!check("Set Current Context For Model Inferencing", true)) return false;

        if (!PreProcess()) return false;
        if (!Inference()) return false;
        if (!PostProcess()) return false;

        return true;
    }


    TimeInfoSessionCore AclSession::GetTimeInfo()
    {
        return *timeInfoPtr;
    }
}