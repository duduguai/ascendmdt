# pragma once
#include <vector>
#include <memory>
#include <optional>

#include "common_types_inner.h"

namespace ZWXT::AI{
    class AclSessionBase
    {
        public:
            class Instance;
            AclSessionBase(const std::string& modelPath, int deviceId, SessionExecParam* paramptr, int batchSizeIn=1, 
                bool needRecordTime=true, bool isaipp=false);
            ~AclSessionBase();
            
            bool Init();
            bool Execute();
        
        private:
            std::unique_ptr<Instance> inst_;
    };

    namespace CV{
        class AclSessionYOLO
        {
            private:
                class Instance;
                std::unique_ptr<Instance> inst_;
            
            public:
                AclSessionYOLO(const std::string& modelPath, int deviceId, SessionExecParam* paramptr, int batchSizeIn=1, 
                bool needRecordTime=true, bool isaipp=false, bool isdvpp=false, bool isnmsop=false, std::optional<int> imwidth=std::nullopt,
                    std::optional<int> imheight=std::nullopt, std::optional<double> lettpad=std::nullopt);
                ~AclSessionYOLO();
                
                bool Init();
                bool Execute();

                // DVPP 引入特定方法
                bool JpegDecodeViaDvpp();
                bool PreprocessViaDvpp();

                // NMS OP 引入特定方法
                bool InitForNmsOp(int clsNum, int anchorsNum, int64_t maxDetsPerCla, float confTh, float iouTh);

                void UpdateAddrParam(int index);
        };
    }
}