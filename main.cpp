#include <algorithm>
#include <string>
#include <stdexcept>
#include "models_yolo.h"

std::vector<std::string> catsname{
    "person",        "bicycle",      "car",           "motorcycle",   "airplane",
    "bus",           "train",        "truck",         "boat",         "traffic light",
    "fire hydrant",  "stop sign",    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",         "sheep",        "cow",
    "elephant",      "bear",         "zebra",         "giraffe",      "backpack",
    "umbrella",      "handbag",      "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",   "kite",         "baseball bat",
    "baseball glove","skateboard",   "surfboard",     "tennis racket","bottle",
    "wine glass",    "cup",          "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",         "sandwich",     "orange",
    "broccoli",      "carrot",       "hot dog",       "pizza",        "donut",
    "cake",          "chair",        "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",            "laptop",       "mouse",
    "remote",        "keyboard",     "cell phone",    "microwave",    "oven",
    "toaster",       "sink",         "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",    "hair drier",   "toothbrush"
};

std::unique_ptr<AI::CV::YOLO::YOLO> GetModel(const std::string& type, int mode, const std::string& modelpath,
        int deviceid, int batchsize, float confth, float iouth, bool isSeg=false)
{
    if (type == "v5") return std::make_unique<AI::CV::YOLO::YOLOv5>(mode, modelpath, deviceid, batchsize, catsname, 640, 640, confth, iouth);
    else if (type == "v5u") return std::make_unique<AI::CV::YOLO::YOLOv5u>(mode, modelpath, deviceid, batchsize, catsname, 640, 640, confth, iouth);
    else if (type == "v6") return std::make_unique<AI::CV::YOLO::YOLOv6>(mode, modelpath, deviceid, batchsize, catsname, 640, 640, confth, iouth);
    else if (type == "v7") return std::make_unique<AI::CV::YOLO::YOLOv7>(mode, modelpath, deviceid, batchsize, catsname, 640, 640, confth, iouth);
    else if (type == "v8") return std::make_unique<AI::CV::YOLO::YOLOv8>(mode, modelpath, deviceid, batchsize, catsname, isSeg, 640, 640, confth, iouth);
    else if (type == "v9") return std::make_unique<AI::CV::YOLO::YOLOv9>(mode, modelpath, deviceid, batchsize, catsname, 640, 640, confth, iouth);
    else if (type == "10") return std::make_unique<AI::CV::YOLO::YOLO10>(mode, modelpath, deviceid, batchsize, catsname, 640, 640, confth);
    else if (type == "11") return std::make_unique<AI::CV::YOLO::YOLO11>(mode, modelpath, deviceid, batchsize, catsname, isSeg, 640, 640, confth, iouth);
    else if (type == "12") return std::make_unique<AI::CV::YOLO::YOLO12>(mode, modelpath, deviceid, batchsize, catsname, 640, 640, confth, iouth);
    else if (type == "13") return std::make_unique<AI::CV::YOLO::YOLO13>(mode, modelpath, deviceid, batchsize, catsname, 640, 640, confth, iouth);
    else if (type == "26") return std::make_unique<AI::CV::YOLO::YOLO26>(mode, modelpath, deviceid, batchsize, catsname, isSeg, 640, 640, confth);
    else if (type == "x") return std::make_unique<AI::CV::YOLO::YOLOX>(mode, modelpath, deviceid, batchsize, catsname, 640, 640, confth, iouth);
    else throw std::invalid_argument("\n[ERROR] 无效 model type. 取值空间 {\"v5\", \"v5u\", \"v6\", \"v7\", \"v8\", \"v9\", \"10\", \"11\", \"12\", \"13\", \"26\", \"x\"}.");
}

int main(int argc, char* argv[])
{
    int mode{0};
    int batchSize{1};
    int deviceid{0};

    float confth{0.25f}, iouth{0.7f};
    bool isSeg{false};

    std::string modeltype;
    std::string modelpath;
    std::string inputpath;
    std::string outputdir;

    for (int i=0; i<argc; ++i)
    {
        std::string arg=argv[i];
        if (arg=="--mode" && i+1<argc) mode = std::stoi(argv[i+1]);
        else if (arg=="--type" && i+1<argc) modeltype=argv[i+1];
        else if (arg=="--model" && i+1<argc) modelpath=argv[i+1];
        else if (arg=="--input" && i+1<argc) inputpath=argv[i+1];
        else if (arg=="--output" && i+1<argc) outputdir=argv[i+1];
        else if (arg=="--bz" && i+1<argc) batchSize=std::stoi(argv[i+1]);
        else if (arg=="--conf" && i+1<argc) confth=std::stof(argv[i+1]);
        else if (arg=="--iou" && i+1<argc) iouth=std::stof(argv[i+1]);
        else if (arg=="--isSeg") isSeg=true;
    }

    if (modelpath.empty() || inputpath.empty()) {
        std::cout << "Usage: " << argv[0]
                  << "--mode [model_runing_mode] --model <om_path> "
                  << "--input <image_or_dir> [--output <dir>]" << std::endl;
        return 1;
    }

    std::transform(modeltype.begin(), modeltype.end(), modeltype.begin(), [](unsigned char c){return static_cast<char>(std::tolower(c));});

    std::unique_ptr<AI::CV::YOLO::YOLO> model = GetModel(modeltype, mode, modelpath, deviceid, batchSize, confth, iouth, isSeg);
    if (!model->Init()) return 1;

    // if (!model->Execute(inputpath, outputdir)) return 1;
    std::vector<std::filesystem::path> inputs(batchSize);
    inputs[0] = std::filesystem::path(inputpath);
    std::vector<AI::CV::YOLO::OutputPerIm> outputs;

    if (!model->Execute(inputs, outputs, true)) return 1;
    return 0;
}