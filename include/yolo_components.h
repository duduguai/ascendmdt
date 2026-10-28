#pragma once
#include <filesystem>
#include <optional>

#include "acl_session.h"

namespace ZWXT::AI::CV{

    class YOLOModel{
        public:
            YOLOModel(){}
            virtual ~YOLOModel(){}

            virtual bool Init() = 0;
            virtual bool Execute(const std::vector<std::filesystem::path>& imFilesPathVec, std::vector<YOLO::Detection>& outPreds,
                std::optional<std::string> saveDir=std::nullopt, bool verbose=true) = 0;
    };


    class YOLOEnd2end : public YOLOModel
    {
        protected:
            class Instance;
            std::unique_ptr<Instance> inst_;

        public:
            YOLOEnd2end(int mode, const std::string& modelpath, int deviceid, std::string modelname, std::optional<std::vector<std::string>> catsname=std::nullopt, 
                int batchsize=1, int imwidth=640, int imheight=640, float confth=0.25, int maxdets=300, int anchors_num=300, int letterboxpadvalue=114, 
                bool needRecorTime=true, bool needdetailinfo=true);
            ~YOLOEnd2end();

            bool Init() override;            
            bool Execute(const std::vector<std::filesystem::path>& imFilesPathVec, std::vector<YOLO::Detection>& outPreds, 
                std::optional<std::string> saveDir=std::nullopt, bool verbose=true) override;
    };

    class YOLOAnchorFree : public YOLOModel
    {
        protected:
            class Instance;
            std::unique_ptr<Instance> inst_;

        public:
            YOLOAnchorFree(int mode, const std::string& modelpath, int deviceid, std::string modelname, std::optional<std::vector<std::string>> catsname=std::nullopt, 
                int batchsize=1, int imwidth=640, int imheight=640, int classesnum=80, float confth=0.25, float iouth=0.7, int maxdets=300,
                int letterboxpadvalue=114, bool needRecorTime=true, bool needdetailinfo=true);
            ~YOLOAnchorFree();

            bool Init() override;
            bool Execute(const std::vector<std::filesystem::path>& imFilesPathVec, std::vector<YOLO::Detection>& outPreds, 
                std::optional<std::string> saveDir=std::nullopt, bool verbose=true) override;
    };

    class YOLOAnchorBase : public YOLOModel
    {
        protected:
            class Instance;
            std::unique_ptr<Instance> inst_;

        public:
            YOLOAnchorBase(int mode, const std::string& modelpath, int deviceid, std::string modelname, std::optional<std::vector<std::string>> catsname=std::nullopt,
                int batchsize=1, int imwidth=640, int imheight=640, int classesnum=80, float confth=0.25, float iouth=0.7, int anchorsfactor=3,
                int maxdets=300, int letterboxpadvalue=114, bool isyolox=false, bool needRecorTime=true, bool needdetailinfo=true);
            ~YOLOAnchorBase();

            bool Init() override;
            bool Execute(const std::vector<std::filesystem::path>& imFilsPathVec, std::vector<YOLO::Detection>& outPreds,
                std::optional<std::string> saveDir=std::nullopt, bool verbose=true) override;
    };
}