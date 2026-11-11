#include "models_yolo_base.h"
#include "models_yolo_list.h"

namespace AI::CV::YOLO
{
    /* YOLO 模型基类 */
    YOLO::~YOLO()
    {
        std::cout<<"[INFO] Model { "<<core->name<<"_"<<core->id<<" } UnLoading."<<std::endl;
    }

    bool YOLO::Init()
    {
        if (!core)
        {
            std::cout<<"[ERROR] Init, Model Core is Null."<<std::endl;
            return false;
        }

        auto timeIni0 = std::chrono::steady_clock::now();

        if (!core->Init())
        {
            std::cout<<"[ERROR] Model { "<<core->name<<"_"<<core->id<<" } Initializing Failed!"<<std::endl;
            return false;
        }

        auto timeIni1 = std::chrono::steady_clock::now();
        float timeIni = std::chrono::duration<float, std::milli>(timeIni1-timeIni0).count();

        std::cout<<"[INFO] Model { "<<core->name<<"_"<<core->id<<" } Initialized Successfully! Costed "<< timeIni
                 <<" ms."<<std::endl;
        
        return true;
    }

    bool YOLO::Execute(const std::string& input, std::optional<std::string> saveDir, bool verbose)
    {
        if (!core)
        {
            std::cout<<"[ERROR] Execute, Model Core is Null."<<std::endl;
            return false;
        }
        return core->Execute(input, saveDir, verbose);
    }

    bool YOLO::Execute(std::vector<std::filesystem::path> imBatchPaths, std::vector<OutputPerIm>& outputs, bool verbose)
    {
        if (!core)
        {
            std::cout<<"[ERROR] Execute, Model Core is Null."<<std::endl;
            return false;
        }
        return core->Execute(imBatchPaths, outputs, verbose);
    }

    /* YOLOv5 */
    YOLOv5::YOLOv5(int mode, const std::string& modelpath, int deviceid, int batchsize, std::optional<std::vector<std::string>> catsname, 
        int imwidth, int imheight, float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad)
    {
        core = std::make_unique<AnchorBasedCore>(mode, modelpath, "YOLOv5", deviceid, batchsize, imwidth, imheight, confTh,
                iouTh, recordtime, maxdets, clsnum, lettpad, 3, catsname, false);
    }
    
    /* YOLOv5u */
    YOLOv5u::YOLOv5u(int mode, const std::string& modelpath, int deviceid, int batchsize, std::optional<std::vector<std::string>> catsname, 
        int imwidth, int imheight, float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad)
    {
        core = std::make_unique<AnchorFreeCore>(mode, modelpath, "YOLOv5u", deviceid, batchsize, imwidth, imheight,
                confTh, iouTh, recordtime, maxdets, clsnum, lettpad, catsname);
    }

    /* YOLOv6 */
    YOLOv6::YOLOv6(int mode, const std::string& modelpath, int deviceid, int batchsize, std::optional<std::vector<std::string>> catsname, 
        int imwidth, int imheight, float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad)
    {
        core = std::make_unique<AnchorBasedCore>(mode, modelpath, "YOLOv6", deviceid, batchsize, imwidth, imheight, confTh, 
                iouTh, recordtime, maxdets, clsnum, lettpad, 1, catsname, false);
    }

    /* YOLOv7 */
    YOLOv7::YOLOv7(int mode, const std::string& modelpath, int deviceid, int batchsize, std::optional<std::vector<std::string>> catsname, 
        int imwidth, int imheight, float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad)
    {
        core = std::make_unique<AnchorBasedCore>(mode, modelpath, "YOLOv7", deviceid, batchsize, imwidth, imheight, confTh, 
            iouTh, recordtime, maxdets, clsnum, lettpad, 3, catsname, false);
    }

    /* YOLOv8 */
    YOLOv8::YOLOv8(int mode, const std::string& modelpath, int deviceid, int batchsize, std::optional<std::vector<std::string>> catsname, 
        bool isSeg, int imwidth, int imheight, float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad)
    {
        core = std::make_unique<AnchorFreeCore>(mode, modelpath, "YOLOv8", deviceid, batchsize, imwidth, imheight, confTh, 
                iouTh, recordtime, maxdets, clsnum, lettpad, catsname, isSeg);
    }

    /* YOLOv9 */
    YOLOv9::YOLOv9(int mode, const std::string& modelpath, int deviceid, int batchsize, std::optional<std::vector<std::string>> catsname, 
        int imwidth, int imheight, float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad)
    {
        core = std::make_unique<AnchorFreeCore>(mode, modelpath, "YOLOv9", deviceid, batchsize, imwidth, imheight, confTh, 
                iouTh, recordtime, maxdets, clsnum, lettpad, catsname);
    }

    /* YOLO10 */
    YOLO10::YOLO10(int mode, const std::string& modelpath, int deviceid, int batchsize, std::optional<std::vector<std::string>> catsname,
                int imwidth, int imheight, float confTh, int anchorsnum, bool recordtime, int maxdets, int lettpad)
    {
        core = std::make_unique<EndToEndCore>(mode, modelpath, "YOLO10", deviceid, batchsize, imwidth, imheight, confTh, 
                anchorsnum, recordtime, maxdets, lettpad, catsname);
    }

    /* YOLO11 */
    YOLO11::YOLO11(int mode, const std::string& modelpath, int deviceid, int batchsize, std::optional<std::vector<std::string>> catsname, 
        bool isSeg, int imwidth, int imheight, float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad)
    {
        core = std::make_unique<AnchorFreeCore>(mode, modelpath, "YOLO11", deviceid, batchsize, imwidth, imheight, confTh, 
                iouTh, recordtime, maxdets, clsnum, lettpad, catsname, isSeg);
    }

    /* YOLO12 */
    YOLO12::YOLO12(int mode, const std::string& modelpath, int deviceid, int batchsize, std::optional<std::vector<std::string>> catsname, 
        int imwidth, int imheight, float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad)
    {
        core = std::make_unique<AnchorFreeCore>(mode, modelpath, "YOLO12", deviceid, batchsize, imwidth, imheight, confTh, 
                iouTh, recordtime, maxdets, clsnum, lettpad, catsname);
    }

    /* YOLO13 */
    YOLO13::YOLO13(int mode, const std::string& modelpath, int deviceid, int batchsize, std::optional<std::vector<std::string>> catsname, 
        int imwidth, int imheight, float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad)
    {
        core = std::make_unique<AnchorFreeCore>(mode, modelpath, "YOLO13", deviceid, batchsize, imwidth, imheight, confTh, 
                iouTh, recordtime, maxdets, clsnum, lettpad, catsname);
    }

    /* YOLO26 */
    YOLO26::YOLO26(int mode, const std::string& modelpath, int deviceid, int batchsize, std::optional<std::vector<std::string>> catsname,
                bool isSeg, int imwidth, int imheight, float confTh, int anchorsnum, bool recordtime, int maxdets, int lettpad)
    {
        core = std::make_unique<EndToEndCore>(mode, modelpath, "YOLO26", deviceid, batchsize, imwidth, imheight, confTh, 
                anchorsnum, recordtime, maxdets, lettpad, catsname, isSeg);
    }

    /* YOLOX */
    YOLOX::YOLOX(int mode, const std::string& modelpath, int deviceid, int batchsize, std::optional<std::vector<std::string>> catsname, 
        int imwidth, int imheight, float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad)
    {
        core = std::make_unique<AnchorBasedCore>(mode, modelpath, "YOLOX", deviceid, batchsize, imwidth, imheight, confTh, 
                iouTh, recordtime, maxdets, clsnum, lettpad, 1, catsname, true);
    }

}