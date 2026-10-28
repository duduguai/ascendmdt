#include <algorithm>
#include <iostream>
#include <memory>

#include "yolo_components.h"
#include "common_types.h"
#include "acl/acl.h"

bool IsImageFile(const std::filesystem::path& path) {
    if (!path.has_extension()) {
        return false;
    }

    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
        ext == ".webp" || ext == ".tif" || ext == ".tiff";
}

std::vector<std::filesystem::path> CollectImages(const std::filesystem::path& input) {
    std::vector<std::filesystem::path> files;
    if (std::filesystem::is_regular_file(input) && IsImageFile(input)) {
        files.push_back(input);
        return files;
    }
    if (!std::filesystem::is_directory(input)) {
        return files;
    }

    for (const auto& entry : std::filesystem::recursive_directory_iterator(input)) {
        if (entry.is_regular_file() && IsImageFile(entry.path())) {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

std::filesystem::path ResolveImageInputPath(const std::filesystem::path& input) {
    if (std::filesystem::is_regular_file(input)) {
        return input;
    }
    if (!std::filesystem::is_directory(input)) {
        return input;
    }

    const auto images_dir = input / "images";
    if (std::filesystem::is_directory(images_dir)) {
        return images_dir;
    }
    return input;
}

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


int main(int argc, char* argv[])
{
    int mode{0};
    int modeltype{0};
    int batchSize{1};
    int anchorfactor{3};
    bool needDetailInfo{true};
    bool isyolox{false};
    std::string modelpath;
    std::string inputpath;
    std::string outputdir;

    for (int i=0; i<argc; ++i)
    {
        std::string arg=argv[i];
        if (arg=="--mode" && i+1<argc) mode = std::stoi(argv[i+1]);
        else if (arg=="--model" && i+1<argc) modelpath=argv[i+1];
        else if (arg=="--input" && i+1<argc) inputpath=argv[i+1];
        else if (arg=="--output" && i+1<argc) outputdir=argv[i+1];
        else if (arg=="--without-detail-info") needDetailInfo=false;
        else if (arg=="--bz") batchSize=std::stoi(argv[i+1]);
        else if (arg=="--modeltype" && i+1<argc) modeltype=std::stoi(argv[i+1]);
        else if (arg=="--isyolox") isyolox=true;
        else if (arg=="--anchorfactor" && i+1<argc) anchorfactor=std::stoi(argv[i+1]);
    }

    if (modelpath.empty() || inputpath.empty()) {
        std::cout << "Usage: " << argv[0]
                  << "--mode [model_runing_mode] --model <om_path> "
                  << "--input <image_or_dir> [--output <dir>]" << std::endl;
        return 1;
    }
    
    // 解析输入图像数据
    const auto resolved_inputpath = ResolveImageInputPath(inputpath);
    std::vector<std::filesystem::path> im_paths = CollectImages(resolved_inputpath);
    if (im_paths.empty())
    {
        std::cerr<<"[ERROR] No images found in: " << resolved_inputpath.string() << std::endl;
        return 1;
    }

    ZWXT::AI::CV::YOLOAnchorFree model1(mode, modelpath, 0, "YOLO11", catsname, batchSize);
    if (!model1.Init()) return 1;

    std::vector<ZWXT::AI::CV::YOLO::Detection> outPreds;

    model1.Execute(im_paths, outPreds, outputdir);

    if (true)
    {
        ZWXT::AI::CV::YOLOAnchorFree model2(mode, modelpath, 0, "YOLO11", catsname, batchSize);
        if (!model2.Init()) return 1;

        // model2.Execute(im_paths, outPreds, outputdir);
    }

    model1.Execute(im_paths, outPreds, outputdir);

    // std::unique_ptr<ZWXT::AI::CV::YOLOModel> model;

    // if (modeltype == 0)
    // {
    //     model = std::make_unique<ZWXT::AI::CV::YOLOEnd2end>(mode, modelpath, 0, catsname, batchSize);
    // }
    // else if (modeltype == 1)
    // {
    //     model = std::make_unique<ZWXT::AI::CV::YOLOAnchorFree>(mode, modelpath, 0, catsname, batchSize);
    // }
    // else
    // {
    //     model = std::make_unique<ZWXT::AI::CV::YOLOAnchorBase>(mode, modelpath, 0, catsname, batchSize, 640, 640, 80, 0.25f, 0.7f, anchorfactor, 300, 114,
    //         isyolox, true, needDetailInfo);
    // }
    
    // if (!model->Init()) return 1;

    // // 解析输入图像数据
    // const auto resolved_inputpath = ResolveImageInputPath(inputpath);
    // std::vector<std::filesystem::path> im_paths = CollectImages(resolved_inputpath);
    // if (im_paths.empty())
    // {
    //     std::cerr<<"[ERROR] No images found in: " << resolved_inputpath.string() << std::endl;
    //     return 1;
    // }

    // std::vector<ZWXT::AI::CV::YOLO::Detection> outPreds;

    // model->Execute(im_paths, outPreds, outputdir);
}
