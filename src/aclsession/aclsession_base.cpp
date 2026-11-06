#include <iostream>
#include "aclsession_base.h"

namespace AI
{
    bool AclSessionBase::CheckTensorConsist(int requireNum, int mode)
    {
        int actualTensorNum;

        if (mode == 0) actualTensorNum = paramPtr->dataIn.data->size();
        else actualTensorNum = paramPtr->dataOut.data->size();

        if (actualTensorNum != requireNum)
        {
            std::cout<<"[ERROR] "+ioType[mode]+", SessionExecParam 与模型要求的 Tensor 数量不同!\n"
                     <<"        SessionExecParam 中 Tensor 数量: "<<actualTensorNum<<",\n"
                     <<"        模型要求输入的 Tensor 数量:       "<<requireNum<<"."<<std::endl;
            return false;
        }
        return true;
    }

    bool AclSessionBase::CheckBytesConsist(int bufferSz, int mode, int idx)
    {
        int actualBytesNum;

        if (mode == 0) actualBytesNum = paramPtr->dataIn.data->at(idx).size()*sizeof(float);
        else actualBytesNum = paramPtr->dataOut.data->at(idx).size()*sizeof(float);

        if (actualBytesNum != bufferSz)
        {
            std::cout<<"[ERROR] "+ioType[mode]+", SessionExecParam 中 Tensor 数据量与模型要求不同! \n"
                     <<"        Tensor 序号:         "<<idx<<", \n"
                     <<"        Param Tensor 数据量: "<<actualBytesNum<<" Bytes, \n"
                     <<"        模型要求数据量:       "<<bufferSz<<" Bytes."<<std::endl;
            return false;
        }
        return true;
    }

    bool AclSessionBase::PrepareBuffer(int idx, void*& buffer, int bufferSz, int mode)
    {
        if (runMode == ACL_HOST)
        {
            ret = aclrtMalloc(&buffer, bufferSz, ACL_MEM_MALLOC_HUGE_FIRST);
            if (!check("Mallo for Model Session", true)) return false;
        }
        else
        {
            if (mode == 0) buffer = paramPtr->dataIn.data->at(idx).data();
            else buffer = paramPtr->dataOut.data->at(idx).data();
        }
        return true;
    }

    bool AclSessionBase::Host2DeviceEP()
    {
        if (runMode == ACL_HOST)
        {
            for (int i=1; i<inputDataInfos.size(); ++i)
            {
                ret = aclrtMemcpy(inputDataInfos[i].buffer, inputDataInfos[i].size, paramPtr->dataIn.data->at(i).data(),
                            inputDataInfos[i].size, ACL_MEMCPY_HOST_TO_DEVICE);
                
                if (!check("Host Input Transfer to Device", true)) return false;
            }
        }

        return true;
    }

    bool AclSessionBase::Device2HostEP()
    {
        if (runMode == ACL_HOST)
        {
            for (int i=1; i<outputDataInfos.size(); ++i)
            {
                ret = aclrtMemcpy(paramPtr->dataOut.data->at(i).data(), outputDataInfos[i].size, outputDataInfos[i].buffer,
                            outputDataInfos[i].size, ACL_MEMCPY_DEVICE_TO_HOST);

                if (!check("Device Output Transfer to Host", true)) return false;
            }
        }

        return true;
    }

    AclSessionBase::AclSessionBase(const std::string& modelpath, int deviceid, SessionParamExec<BaseIn, BaseOut>* paramptr, 
                int batchsize, bool needrecordtime, bool isaipp):
                AclSession(modelpath, deviceid, batchsize, needrecordtime), isAipp(isaipp), paramPtr(paramptr)
                {}
    AclSessionBase::~AclSessionBase(){}
}