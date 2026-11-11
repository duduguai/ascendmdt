#pragma once
#include "aclsession_core.h"
#include "types_session_general.h"

namespace AI
{
    class AclSessionGeneral: public AclSession
    {
        private:
            // 是否启用 AIPP 功能标志位
            bool isAipp{false};

            // Session 推理执行所需参数
            SessionParamExec<BaseIn, BaseOut>* paramPtr{nullptr};

            bool CheckBytesConsist(int bufferSz, int mode, int idx) override;
            
            bool CheckTensorConsist(int requireNum, int mode) override;

            bool PrepareBuffer(int idx, void*& buffer, int bufferSz, int mode) override;

            bool Host2DeviceEP() override;

            bool Device2HostEP() override;

        public:
            AclSessionGeneral(const std::string& modelpath, int deviceid, SessionParamExec<BaseIn, BaseOut>* paramptr, 
                int batchsize, bool needrecordtime, bool isaipp);
            virtual ~AclSessionGeneral();
    };
}