#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "yolo_components.h"
#include "common_types_inner.h"


namespace ZWXT::AI::CV{


    namespace{
        const std::vector<int> colors = {235, 219, 11, 243, 243, 243, 183, 223, 0, 221, 111, 255, 0, 237, 204, 68, 243, 0, 255, 255, 0, 179, 255, 1,
        11, 255, 162, 255, 42, 4, 79, 68, 255, 255, 0, 189, 255, 180, 0, 186, 0, 221, 0, 192, 38, 255, 36, 125, 104, 0, 123, 108, 
        27, 255, 47, 109, 252, 104, 31, 17};

        // 动态显示当前进度
        // current -> 当前完成的进度数
        // total -> 该任务总的进度数
        // barWidth -> 进度条的显示长度
        void showProgress(int current, int total, int barWidth=50)
        {
            float progress = static_cast<float>(current)/static_cast<float>(total);
            int pos = static_cast<int>(barWidth*progress);

            std::cout<< "\rInferencing ... [";
            for (int i = 0; i<barWidth; ++i)
            {
                if (i<pos) std::cout<<"=";
                else if (i==pos) std::cout<<">";
                else std::cout<<" ";
            }
            std::cout<<"]"<<std::setw(3)<<static_cast<int>(progress*100)<<"%"<<"  "<<"("<<current<<"/"<<total<<")";
            std::cout<<std::flush;
        }

        // 获取当前输入的图像文件集根据给定的 BatchSize 能够分为多少 Epoch
        // imPathsVec -> 输入图像文件路径集
        // batchSize -> 批大小
        int getEpoches(const std::vector<std::filesystem::path> imPathsVec, int batchSize)
        {
            int epoches, totalFilesNum;
            totalFilesNum = imPathsVec.size();
            if (totalFilesNum % batchSize == 0) epoches = totalFilesNum / batchSize;
            else epoches = totalFilesNum / batchSize + 1;
            
            return epoches;
        }

        // 加载原始二进制图像文件
        // imPath -> 图像文件路径
        // imBin -> 输出的图像二进制数据
        // size -> 输出的图像二进制数据的大小
        bool loadImBinary(const std::string& imPath, std::vector<uint8_t>& imBin, size_t& size)
        {
            // 打开文件
            FILE* fp = fopen(imPath.c_str(), "rb");
            if (!fp)
            {
                std::cerr<<"[WARNING] 无法打开图像文件："<<imPath<<"! Skip!"<<std::endl;
                return false;
            }

            // 获取文件大小
            fseek(fp, 0, SEEK_END);
            size = ftell(fp);
            fseek(fp, 0, SEEK_SET);

            // 读取数据
            imBin.resize(size);
            size_t readSize = fread(imBin.data(), 1, size, fp);
            fclose(fp);

            // 检验读取是否正确
            if (readSize != size)
            {
                std::cerr<<"[WARNING] 图像文件读取错误: "<<imPath<<"! Skip!"<<std::endl;
                return false;
            }

            return true;
        }

        // 通过OpenCV加载图像文件
        // imPath -> 图像文件路径
        // imMat -> 输出 cv::Mat 格式图像数据
        bool loadImMat(const std::string& imPath, cv::Mat& imMat)
        {
            cv::Mat im = cv::imread(imPath, cv::IMREAD_COLOR);
            if (im.empty()) 
            {   
                std::cerr<<"[WARNING] 图像文件无法正常加载: "<<imPath<<"! Skip!"<<std::endl;
                return false;
            }
            imMat = std::move(im);

            return true;
        }

        // 按照置信度为 BoundingBox 排序
        // a, b -> BoundingBox
        bool SortByScore(const YOLO::BoundingBox & a, const YOLO::BoundingBox & b)
        {
            return a.score > b.score;
        }

        // 计算两个 BoundingBox 的交并比
        // box1, box2 -> BoundingBox
        float ComputeIoU(const YOLO::BoundingBox& box1, const YOLO::BoundingBox& box2)
        {
            float area_1 = (box1.x2-box1.x1)*(box1.y2-box1.y1);
            float area_2 = (box2.x2-box2.x1)*(box2.y2-box2.y1);
            float inter_x1 = std::max(box1.x1, box2.x1);
            float inter_y1 = std::max(box1.y1, box2.y1);
            float inter_x2 = std::min(box1.x2, box2.x2);
            float inter_y2 = std::min(box1.y2, box2.y2);
            
            float inter_area = static_cast<float>(std::max(inter_x2-inter_x1, 0.0f))*static_cast<float>(std::max(inter_y2-inter_y1, 0.0f));
            float union_area = area_1 + area_2 - inter_area;
            return inter_area / (union_area + 1e-12f);
        }


        class YOLOBase{
            public:
            // 模型名称
                std::string modelName{""};
            // 模型ID
                int modelId{-1};

            protected:
            // 创建实例统计量
                inline static int _count = 0;

            // 是否记录各阶段耗时
                bool isRecordTime{true};
                bool needDetailInfo{true};
            //  AIPP 标志位
                bool isAipp{false};
            //  DVPP 标志位
                bool isDvpp{false};
            //  NMS OP 标志位
                bool isNmsop{false};

            // ACL 推理会话
                std::unique_ptr<AclSessionYOLO> session{nullptr};

            // 原始图像数据
                std::unique_ptr<std::vector<cv::Mat>> imVecMat{nullptr};

                std::vector<bool> imValid;
                std::vector<int> imValidIdx;

            // LetterBox 处理过程产生的参数，将输出框缩放到原图尺寸的关键
                std::unique_ptr<std::vector<LetterBoxMeta>> imMeta;

            // 模型输入相关参数
                int batchSize{-1};
                int imWidth{-1};
                int imHeight{-1};
                // 输入进模型的每张图像单通道像素量
                size_t numsPerChannel{0};
                // 输入进模型的每张图像的像素量
                size_t numsPerIm{0};
                // DVPP 默认为 Batch 中每条读取的二进制数据 reserve 保留的内存空间大小 (2MB)
                int defauSzDvpp{2*1024*1024};
        
            // 模型输出相关参数
                int boxCordsNum{4};
                int clsNum{0};
                int scoreNum{0};
                int anchorsNum{-1};
            
            // 模型输出的类别信息
                std::vector<std::string> catsName;
            
            // 后处理相关参数
                float confTh{0.0f};
                float iouTh{0.0f};
                int maxDets{-1};
            
            // LetterBox相关参数
                int letterboxPadvalue{-1};
            
            // Execute 输入参数
                SessionExecParam param;

            // NMS OP 状态下所需要的常量
                std::unique_ptr<std::vector<int>> boxStridePtr{nullptr};
                std::unique_ptr<std::vector<int>> scoreStridePtr{nullptr};

            private:
            // 数据前处理

                // 通过OpenCV将图像文件路径 std::vector<std::filesystem::path> 加载为 std::vector<cv::Mat>
                // imPathsVec -> 保存所有需要推理的图像文件的 vector
                // offset -> 当前 Batch 的起始位置相对于 vector 起始位置的偏移量
                bool loadImVec(const std::vector<std::filesystem::path> imPathsVec, int offset, int& validNum)
                {
                    int totalFilesNum = imPathsVec.size();
                    int index = offset;
                    for (int i=0; i<batchSize;)
                    {
                        if (index < totalFilesNum)
                        {
                            if (!isDvpp)
                            {
                                if (loadImMat(imPathsVec[index].string(), imVecMat->at(i))) 
                                {
                                    imValid[i] = true;
                                    imValidIdx[i] = index;
                                    ++validNum;
                                    ++i;
                                }
                            }
                            else
                            {
                                size_t size;
                                if (loadImBinary(imPathsVec[index].string(), param.dvppParam.dataInDvpp->at(i), size))
                                {
                                    if (size >= defauSzDvpp) session->UpdateAddrParam(i);
                                    imValid[i] = true;
                                    imValidIdx[i] = index;
                                    ++validNum;
                                    ++i;
                                }
                            }
                            ++index;
                        }
                        else
                        {
                            if (i==0) return false;
                            if (!isDvpp)
                            {
                                imVecMat->at(i) = imVecMat->at(i-1);
                            }
                            else
                            {
                                param.dvppParam.dataInDvpp->at(i) = param.dvppParam.dataInDvpp->at(i-1);
                            }
                            imValid[i] = false;
                            ++i;
                        }
                    }
                    return true;
                }

                // 单次调用完成单张图像通过OpenCV实现LetterBox
                // imBGR -> 通过 OpenCV 加载图像文件得到的原始 cv::Mat
                // letterboxedBGR -> 经过 LetterBox 后的输出数据
                // meta -> 维护 LetterBox 过程中使用的缩放、填充参数
                void LetterBox(const cv::Mat& imBGR, cv::Mat& letterboxedBGR, LetterBoxMeta& meta)
                {
                    meta.imOriW = imBGR.cols;
                    meta.imOriH = imBGR.rows;
                    const float scale_w = static_cast<float>(imWidth) / static_cast<float>(imBGR.cols);
                    const float scale_h = static_cast<float>(imHeight) / static_cast<float>(imBGR.rows);
                    meta.scale = std::min(scale_w, scale_h);

                    const int resized_w = static_cast<int>(std::round(static_cast<float>(imBGR.cols)*meta.scale));
                    const int resized_h = static_cast<int>(std::round(static_cast<float>(imBGR.rows)*meta.scale));
                    const float dw = static_cast<float>(imWidth - resized_w) * 0.5f;
                    const float dh = static_cast<float>(imHeight - resized_h) * 0.5f;
                    meta.left = static_cast<int>(std::round(dw - 0.1f));
                    meta.top = static_cast<int>(std::round(dh - 0.1f));
                    meta.right = static_cast<int>(std::round(dw + 0.1f));
                    meta.bottom = static_cast<int>(std::round(dh + 0.1f));

                    cv::Mat resized_bgr;
                    cv::resize(imBGR, resized_bgr, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

                    cv::copyMakeBorder(
                        resized_bgr,
                        letterboxedBGR,
                        meta.top,
                        meta.bottom,
                        meta.left,
                        meta.right,
                        cv::BORDER_CONSTANT,
                        cv::Scalar(letterboxPadvalue, letterboxPadvalue, letterboxPadvalue)
                    );
                }

                // 单次调用完成单张图像数据 1) uint8 -> float; 2) BGR -> RGB; 3) /255.0 归一化
                // letterboxedBGR -> 经过 LetterBox 后的输出数据
                // imIn -> 经过 Normalize 后的输出数据
                // batchIdx -> 该数据位于当前 Batch 中的序号
                // isYOLOX -> 是否为 YOLOX，YOLOX 模型的预处理中没有 /255.0 归一化步骤
                void Normalize(const cv::Mat& letterboxedBGR, std::vector<float>& imIn, int batchIdx, std::optional<bool> isYOLOX=std::nullopt)
                {
                    float inv;
                    if (isYOLOX && *isYOLOX) inv = 1.0f;
                    else inv = 1.0f/255.0f;

                    float* c0 = imIn.data() + 0*numsPerChannel + batchIdx*numsPerIm;
                    float* c1 = imIn.data() + 1*numsPerChannel + batchIdx*numsPerIm;
                    float* c2 = imIn.data() + 2*numsPerChannel + batchIdx*numsPerIm;

                    for (uint32_t y=0; y<imHeight; ++y)
                    {
                        const uint8_t* row = letterboxedBGR.ptr<uint8_t>(static_cast<int>(y));
                        for (uint32_t x=0; x<imWidth; ++x)
                        {
                            std::size_t idx = static_cast<std::size_t>(y)*static_cast<std::size_t>(imWidth)+static_cast<std::size_t>(x);
                            int rowIdx = x*3;
                            c0[idx] = static_cast<float>(row[rowIdx+2])*inv;
                            c1[idx] = static_cast<float>(row[rowIdx+1])*inv;
                            c2[idx] = static_cast<float>(row[rowIdx])*inv;
                        }
                    }
                }

                // 单次运行将 BatchSize 个 cv::Mat 格式的图像数据通过 OpenCV 实现数据前处理
                void PreprocViaOpenCV(std::optional<bool> isYOLOX=std::nullopt)
                {
                    std::chrono::steady_clock::time_point timeLett0, timeLett1, timeNorm0, timeNorm1;
                    int offset{0};
                    for (int i=0; i<batchSize; ++i)
                    {
                        cv::Mat imLett;
                        if (isRecordTime && needDetailInfo && imValid[i]) timeLett0 = std::chrono::steady_clock::now();
                        LetterBox(imVecMat->at(i), imLett, imMeta->at(i));
                        if (isRecordTime && needDetailInfo && imValid[i]) 
                        {
                            timeLett1 = std::chrono::steady_clock::now();
                            param.timeInfo->let += std::chrono::duration<float, std::milli>(timeLett1-timeLett0).count();
                        }
                        
                        // 对于非 AIPP 通过 OpenCV 执行归一化和标准化操作，并将数据转移到 Session 输入参数中
                        if (!isAipp)
                        {   
                            if (isRecordTime && needDetailInfo && imValid[i]) timeNorm0 = std::chrono::steady_clock::now();
                            Normalize(imLett, param.baseParam.dataInBase->at(0), i, isYOLOX);
                            if (isRecordTime && needDetailInfo && imValid[i]) 
                            {
                                timeNorm1 = std::chrono::steady_clock::now();
                                param.timeInfo->nor += std::chrono::duration<float, std::milli>(timeNorm1-timeNorm0).count();
                            }
                            continue;
                        }

                        // 对于 AIPP 直接将 uint8_t 格式的数据转移到 Session 输入参数中
                        if (!imLett.isContinuous()) imLett = imLett.clone();
                        std::memcpy(param.aippParam.dataInAipp->at(0).data()+offset, imLett.data, numsPerIm);
                        offset += numsPerIm;
                    }
                }

                // 将预测框缩放到原图的尺寸上
                // outBoxes -> 模型对单张图片预测输出的所有框集合
                // meta -> 图像在 LetterBox 过程中保存的信息
                void ScaleToOri(std::vector<YOLO::BoundingBox>& outBoxes, const LetterBoxMeta& meta)
                {
                    float inv_scale = meta.scale > 0.0f ? (1.0f / meta.scale) : 1.0f;
                    for (std::size_t idx=0; idx<outBoxes.size();)
                    {
                        outBoxes[idx].x1 = (outBoxes[idx].x1 - static_cast<float>(meta.left)) * inv_scale;
                        outBoxes[idx].y1 = (outBoxes[idx].y1 - static_cast<float>(meta.top)) * inv_scale;
                        outBoxes[idx].x2 = (outBoxes[idx].x2 - static_cast<float>(meta.left)) * inv_scale;
                        outBoxes[idx].y2 = (outBoxes[idx].y2 - static_cast<float>(meta.top)) * inv_scale;

                        outBoxes[idx].x1 = std::clamp(outBoxes[idx].x1, 0.0f, static_cast<float>(meta.imOriW - 1));
                        outBoxes[idx].y1 = std::clamp(outBoxes[idx].y1, 0.0f, static_cast<float>(meta.imOriH - 1));
                        outBoxes[idx].x2 = std::clamp(outBoxes[idx].x2, 0.0f, static_cast<float>(meta.imOriW - 1));
                        outBoxes[idx].y2 = std::clamp(outBoxes[idx].y2, 0.0f, static_cast<float>(meta.imOriH - 1));
                        if (outBoxes[idx].x1>=outBoxes[idx].x2 || outBoxes[idx].y1 >= outBoxes[idx].y2)
                        {
                            outBoxes.erase(outBoxes.begin() + static_cast<long long>(idx));
                        }
                        else
                        {
                            ++idx;
                        }
                    }
                }

                // 将预测输出在原图上进行描框
                // im -> 预测输入的原图，未经预处理
                // bboxes -> 预测输出的 BBox
                void DrawBoundingBox(cv::Mat& im, const std::vector<YOLO::BoundingBox>& bboxes)
                {
                    int im_width = im.cols;
                    for (int i=0; i<bboxes.size(); ++i)
                    {
                        int colorIdx = bboxes[i].cls % 20;
                        int b = colors[3*colorIdx];
                        int g = colors[3*colorIdx+1];
                        int r = colors[3*colorIdx+2];

                        cv::Scalar box_color = cv::Scalar(b, g, r);
                        cv::Scalar text_color;
                        if (colorIdx < 9)
                        {
                            text_color = cv::Scalar(104, 31, 17);
                        }
                        else
                        {
                            text_color = cv::Scalar(255, 255, 255);
                        }

                        float p1_x = bboxes[i].x1;
                        float p1_y = bboxes[i].y1;
                        float p2_x = bboxes[i].x2;
                        float p2_y = bboxes[i].y2;

                        cv::rectangle(
                            im,
                            cv::Point(p1_x, p1_y),
                            cv::Point(p2_x, p2_y),
                            box_color,
                            2,
                            cv::LINE_AA
                        );

                        // 处理 BBox 的标注信息
                        std::string label;
                        std::ostringstream oss;
                        if (!catsName.empty()) oss << catsName.at(bboxes[i].cls)<<": "<< std::fixed << std::setprecision(2) << bboxes[i].score;
                        else oss << bboxes[i].cls<<": "<< std::fixed << std::setprecision(2) << bboxes[i].score;
                        label = oss.str();

                        cv::Size text_size = cv::getTextSize(label, 0, 0.6, 2, 0);
                        text_size.height += 3;
                        bool outside = (p1_y >= (text_size.height));
                        if (p1_x + text_size.width > im_width)
                        {
                            p1_x = im_width - text_size.width;
                        }
                        p2_x = p1_x + text_size.width;
                        if (outside) p2_y = p1_y - text_size.height;
                        else p2_y = p1_y + text_size.height;

                        cv::rectangle(
                            im,
                            cv::Point(p1_x, p1_y),
                            cv::Point(p2_x, p2_y),
                            box_color,
                            -1,
                            cv::LINE_AA
                        );

                        if (outside) p1_y = p1_y - 2;
                        else p1_y = p1_y + text_size.height - 1;

                        cv::putText(
                            im,
                            label,
                            cv::Point(p1_x, p1_y),
                            0,
                            0.6,
                            text_color,
                            2,
                            cv::LINE_AA
                        );
                    }
                }

                // 通过 OpenCV 已加载的原始图像数据进行描框图像保存
                // saveDir -> 保存输出图像的目录
                // det -> 经模型推理输出的结果
                // batchIdx -> 当前输出位于 Batch 中的序号
                void saveViaMat(const std::string& saveDir, const YOLO::Detection& det, int batchIdx)
                {
                    cv::Mat im = imVecMat->at(batchIdx);
                    DrawBoundingBox(im, det.box);
                    std::string savePath = saveDir + "/" + det.filename;
                    cv::imwrite(savePath, im);
                }

                // 通过 OpenCV 读取 JPEG 文件获得原始图像数据实现描框图像保存
                // saveDir -> 保存输出图像的目录
                // det -> 经模型推理输出的结果
                // filepath -> 当前输出数据对应的原始数据的图像文件路径
                void saveViaFile(const std::string& saveDir, const YOLO::Detection& det, const std::string& filepath)
                {
                    cv::Mat im = cv::imread(filepath);
                    DrawBoundingBox(im, det.box);
                    std::string savePath = saveDir + "/" + det.filename;
                    cv::imwrite(savePath, im);
                }

                // 使用 NMS 算子场景下，对 NMS 算子数据进行解码到标准输出格式
                // indicesPtr -> 当前 Batch 中各独立数据经 NMS 输出后 indices 数据的起始指针
                // outBoxes -> 解码输出的标准格式数据
                // numsPerData -> 每份独立数据经 NME 算子输出后 indices 的数据量
                // batchIdx -> Debug 时使用的参数
                void PostDecodeForNmsOp(const int32_t* indicesPtr, std::vector<YOLO::BoundingBox>& outBoxes, int numsPerData, int batchIdx)
                {
                    const float* boxPoi = param.nmsopParam.boxes->data();
                    const float* scorePoi = param.nmsopParam.scores->data();

                    int32_t btIdx, claIdx, boxIdx, scoreOffset, boxOffset;
                    float score, cx, cy, w, h, x1, y1, x2, y2;

                    outBoxes.reserve(numsPerData);
                    for (int i=0; i<numsPerData; ++i)
                    {
                        btIdx = *(indicesPtr);

                        if (btIdx >= 0)
                        {
                            if (btIdx != batchIdx) std::cout<<"[DEBUG] 预期 batchIdx: "<<batchIdx<<"实际读取 batchIdx: "<<btIdx<<std::endl;
                            claIdx = *(indicesPtr+1);
                            boxIdx = *(indicesPtr+2);

                            scoreOffset = batchIdx*scoreStridePtr->at(0) + claIdx*scoreStridePtr->at(1) + boxIdx;
                            boxOffset = batchIdx*boxStridePtr->at(0) + boxIdx*boxStridePtr->at(1);

                            score = *(scorePoi + scoreOffset);
                            cx = *(boxPoi + boxOffset);
                            cy = *(boxPoi + boxOffset + 1);
                            w = *(boxPoi + boxOffset + 2);
                            h = *(boxPoi + boxOffset + 3);

                            x1 = cx - w*0.5f;
                            y1 = cy - h*0.5f;
                            x2 = cx + w*0.5f;
                            y2 = cy + h*0.5f;

                            YOLO::BoundingBox box{x1, y1, x2, y2, score, claIdx};
                            outBoxes.push_back(std::move(box));
                        }

                        indicesPtr += 3;
                    }
                }

            public:
                YOLOBase(int mode, const std::string& modelPath, int deviceId, int batchsize, int imwidth, int imheight, 
                float confth, int maxdets, int letterboxpadvalue, bool needRecorTime, bool needdetailInfo, std::string modelname,
                std::optional<std::vector<std::string>> catsname=std::nullopt): 
                batchSize(batchsize), confTh(confth), maxDets(maxdets), isRecordTime(needRecorTime), needDetailInfo(needdetailInfo),
                modelName(modelname)
                {
                    if (catsname) catsName = std::move(*catsname);

                    imValid.resize(batchsize);
                    imValidIdx.resize(batchsize);
                
                    if (needRecorTime) param.timeInfo = std::make_unique<BatchTimeInfo>();

                    if (mode == 1 || mode == 2 || mode == 4 || mode ==5)
                    {
                        isAipp = true;
                        if (mode == 2 || mode ==5) isDvpp = true;
                    }

                    if (mode == 3 || mode == 4 || mode == 5) isNmsop = true;

                    std::optional<int> iw{std::nullopt}, ih{std::nullopt};
                    std::optional<double> lettpad{std::nullopt};
                    if (!isDvpp) 
                    {
                        imWidth = imwidth; imHeight=imheight; letterboxPadvalue = letterboxpadvalue;
                        numsPerChannel = imwidth*imheight;
                        numsPerIm = 3*numsPerChannel;
                        imMeta = std::make_unique<std::vector<LetterBoxMeta>>(batchSize);
                        imVecMat = std::make_unique<std::vector<cv::Mat>>(batchsize);
                    }
                    else
                    {
                        iw = imwidth; ih = imheight; lettpad = letterboxpadvalue;
                    }

                    session = std::make_unique<AclSessionYOLO>(modelPath, deviceId, &param, batchsize, needdetailInfo, isAipp, isDvpp, isNmsop, iw, ih, lettpad);
                    
                    modelId = _count;
                    _count += 1;
                }
                virtual ~YOLOBase(){}

                // 初始化 Model
                virtual bool Init()
                {
                    // 根据 mode 在 param 中创建对应的参数空间
                    if (!isAipp)
                    {
                        param.baseParam.dataInBase = std::make_unique<std::vector<std::vector<float>>>(1);
                        param.baseParam.dataInBase -> at(0).resize(batchSize*numsPerIm);
                    }
                    else
                    {
                        if (!isDvpp)
                        {  
                            param.aippParam.dataInAipp = std::make_unique<std::vector<std::vector<uint8_t>>>(1);
                            param.aippParam.dataInAipp -> at(0).resize(batchSize*numsPerIm);
                        }
                        else
                        {
                            param.dvppParam.dataInDvpp = std::make_unique<std::vector<std::vector<uint8_t>>>(batchSize);
                            for (int i=0; i<batchSize; ++i) param.dvppParam.dataInDvpp->at(i).reserve(defauSzDvpp);
                            param.dvppParam.letterboxMeta = std::make_unique<std::vector<LetterBoxMeta>>(batchSize);
                        }
                    }

                    int maxDetsPerCls;
                    if (!isNmsop)
                    {
                        param.baseParam.dataOutBase = std::make_unique<std::vector<std::vector<float>>>(1);
                        param.baseParam.dataOutBase -> at(0).resize(batchSize*anchorsNum*(boxCordsNum+clsNum+scoreNum));
                    }
                    else
                    {
                        boxStridePtr = std::make_unique<std::vector<int>>(std::vector<int>{4*anchorsNum, 4});
                        scoreStridePtr = std::make_unique<std::vector<int>>(std::vector<int>{anchorsNum*clsNum, anchorsNum});

                        maxDetsPerCls = (maxDets + clsNum - 1) / clsNum;
                        maxDetsPerCls = (maxDetsPerCls > 15) ? maxDetsPerCls:15;

                        param.nmsopParam.boxes = std::make_unique<std::vector<float>>(batchSize*anchorsNum*boxCordsNum);
                        param.nmsopParam.scores = std::make_unique<std::vector<float>>(batchSize*anchorsNum*clsNum);
                        param.nmsopParam.indices = std::make_unique<std::vector<int32_t>>(batchSize*maxDetsPerCls*clsNum*3);
                    }

                    // 对 session 进行初始化
                    if (! session -> Init()) return false;
                    if (isNmsop){if (! session -> InitForNmsOp(clsNum, anchorsNum, maxDetsPerCls, confTh, iouTh)) return false;}
                    return true;
                }

                // 预留后处理接口
                virtual void PostDecode(const float* ptr, std::vector<YOLO::BoundingBox>& outBoxes){}

                // 模型推理执行入口
                // imFilePathsVec -> 包含所有需要推理的图像文件的 vector
                // preds -> 模型推理后返回的预测输出
                // verbose -> 是否显示推理结果统计信息
                // saveDir -> 描框后图像保存的目录，默认为 nullptr 即不执行保存操作
                bool Execute(const std::vector<std::filesystem::path>& imFilePathsVec, std::vector<YOLO::Detection>& preds, bool verbose,
                std::optional<std::string> saveDir, std::optional<bool> isYOLOX=std::nullopt)
                {
                    // 首先分析传入的图像文件按照设置的 BatchSize 需要多少个 Epoch 才能完成推理
                    if (imFilePathsVec.empty())
                    {
                        std::cerr<<"[ERROR] 传入的图像路径集为空，无法进行推理！"<<std::endl;
                        return false;
                    } 
                    int Epoches = getEpoches(imFilePathsVec, batchSize);

                    // 判断是否需要保存描框结果，若需要保存则判断当前保存目录是否创建
                    if (saveDir && *saveDir != "")
                    {
                        std::filesystem::create_directories(*saveDir);
                    }

                    // 构建时间点记录变量以及统计变量
                    std::chrono::steady_clock::time_point timeLd0, timeLd1, timePre0, timePre1, timeE2e0, timeE2e1, timePos0, timePos1;
                    float pre{0.0f}, e2e{0.0f}, pos{0.0f}, ld{0.0f}, let{0.0f}, nor{0.0f}, h2d{0.0f}, inf{0.0f}, d2h{0.0f}, nms{0.0f}, sto{0.0f};

                    int succInf = 0;
                    int infBoxNum = 0;
                    // 执行推理循环
                    std::cout<<"[INFO] Model {"+modelName+"_"<<modelId<<"} Starts Inferencing."<<std::endl;
                    for (int epoch=0; epoch<Epoches; ++epoch)
                    {
                        if (isRecordTime) param.timeInfo->Init();
                        showProgress(epoch, Epoches);
                        int validFileNum = 0;

                    // 读取图像文件
                        if (isRecordTime && needDetailInfo) timeLd0 = std::chrono::steady_clock::now();
                        if (! loadImVec(imFilePathsVec, epoch*batchSize, validFileNum))
                        {
                            std::cerr<<std::endl<<"[WARNING] 已无法从提供的图像文件中正常加载足够的图像构成一个 Batch ! 退出推理循环！"<<std::endl;
                            break;
                        }
                        if (isDvpp)
                        {
                            if (!session->JpegDecodeViaDvpp())
                            {
                                std::cout<<"[WARNING] 当前 Batch 无法通过 DVPP 实现成功解码, 跳过当前 Batch! Skip!"<<std::endl;
                                continue;
                            }
                        }
                        if (isRecordTime && needDetailInfo) 
                        {
                            timeLd1 = std::chrono::steady_clock::now();
                            param.timeInfo -> ld += std::chrono::duration<float, std::milli>(timeLd1 - timeLd0).count();
                        }
                    // 前处理流程
                        // 对图像数据进行 LetterBox -> BGR to RGB -> /255.0
                        if (isRecordTime) timePre0 = std::chrono::steady_clock::now();
                        // 对于非 DVPP 的数据前处理流程
                        if (!isDvpp)
                        {
                            // 通过 OpenCV 实现图像数据前处理
                            PreprocViaOpenCV(isYOLOX);
                        }
                        else
                        {
                            // 通过 DVPP 实现图像数据前处理
                            if (! session->PreprocessViaDvpp()) continue;
                        }
                        if (isRecordTime)
                        {
                            timePre1 = std::chrono::steady_clock::now();
                            param.timeInfo->pre += std::chrono::duration<float, std::milli>(timePre1-timePre0).count();
                        }
                        
                    // 执行模型推理
                        if (isRecordTime) timeE2e0 = std::chrono::steady_clock::now();
                        if (! session -> Execute()) 
                        {
                            std::cerr<<std::endl<<"[WARNING] Epoch "<<epoch<<" 推理失败，跳过该 Batch 数据的推理."<<std::endl;
                            continue;
                        }
                        if (isRecordTime) {timeE2e1 = std::chrono::steady_clock::now(); param.timeInfo->e2e += std::chrono::duration<float, std::milli>(timeE2e1-timeE2e0).count();}

                    // 后处理流程
                        if (isRecordTime) timePos0 = std::chrono::steady_clock::now();

                        std::chrono::steady_clock::time_point timepos0, timepos1, timepos2, timeSave0, timeSave1;

                        int numsPerData;
                        if (!isNmsop)
                        {
                            numsPerData = (boxCordsNum+clsNum+scoreNum)*anchorsNum;
                            if (param.baseParam.dataOutBase->at(0).size() != numsPerData*batchSize)
                            {
                                std::cerr<<std::endl<<"[WARNING] Epoch "<<epoch<<" 推理输出结果的数据量与预期不符，跳过该 Batch 的后处理流程."<<std::endl;
                                continue;
                            }
                        }
                        else
                        {
                            numsPerData = param.nmsopParam.indices->size() / 3 / batchSize;
                        }
                        
                        const float* ptr;
                        const int32_t* ptrNmsOp;
                        if (!isNmsop) ptr = param.baseParam.dataOutBase->at(0).data();
                        else ptrNmsOp = param.nmsopParam.indices->data();
                        
                        for (int batchIdx=0; batchIdx<batchSize; ++batchIdx)
                        {
                            if (! imValid[batchIdx]) continue;

                            YOLO::Detection det;
                            det.filename = imFilePathsVec[imValidIdx[batchIdx]].filename().string();

                            // 执行 Decode 
                            if (isRecordTime && needDetailInfo) timepos0 = std::chrono::steady_clock::now();
                            if (!isNmsop) PostDecode(ptr, det.box);
                            else PostDecodeForNmsOp(ptrNmsOp, det.box, numsPerData, batchIdx);
                            
                            
                            if (isRecordTime && needDetailInfo) timepos1 = std::chrono::steady_clock::now();
                            
                            // Scale To Original Size
                            if (!isDvpp) ScaleToOri(det.box, imMeta->at(batchIdx));
                            else ScaleToOri(det.box, param.dvppParam.letterboxMeta->at(batchIdx));
                            if (isRecordTime && needDetailInfo)
                            {
                                timepos2 = std::chrono::steady_clock::now();
                                param.timeInfo->nms += std::chrono::duration<float, std::milli>(timepos1-timepos0).count();
                                param.timeInfo->sto += std::chrono::duration<float, std::milli>(timepos2-timepos1).count();
                            }

                            infBoxNum += det.box.size();

                            // 描框并保存图像
                            if (isRecordTime) timeSave0 = std::chrono::steady_clock::now();
                            if (saveDir && *saveDir != "")
                            {
                                if (!isDvpp) saveViaMat(*saveDir, det, batchIdx);
                                else saveViaFile(*saveDir, det, imFilePathsVec[imValidIdx[batchIdx]].string());
                            }
                            if (isRecordTime) 
                            {
                                timeSave1 = std::chrono::steady_clock::now(); 
                                param.timeInfo->pos -= std::chrono::duration<float, std::milli>(timeSave1-timeSave0).count();
                            }

                            // 将检测输出结果保存到输出变量中
                            preds.push_back(std::move(det));

                            if (!isNmsop) ptr += numsPerData;
                            else ptrNmsOp += numsPerData*3;
                        }
                    

                        if (isRecordTime) 
                        {
                            timePos1 = std::chrono::steady_clock::now(); 
                            param.timeInfo->pos += std::chrono::duration<float, std::milli>(timePos1-timePos0).count();

                            pre += param.timeInfo -> pre;
                            e2e += param.timeInfo -> e2e;
                            pos += param.timeInfo -> pos;

                            ld += param.timeInfo -> ld;
                            let += param.timeInfo -> let;
                            nor += param.timeInfo -> nor;
                            h2d += param.timeInfo -> h2d;
                            inf += param.timeInfo -> inf;
                            d2h += param.timeInfo -> d2h;
                            nms += param.timeInfo -> nms;
                            sto += param.timeInfo -> sto;
                        }
                        // 成功推理的图片计数
                        succInf += validFileNum;
                    }
                    showProgress(Epoches, Epoches); std::cout<<std::endl<<std::endl;

                    if (succInf == 0)
                    {
                        std::cerr<<"[ERROR] No images had been inferenced successfully !"<<std::endl;
                        return false;
                    }

                    if (verbose)
                    {
                        std::cout<<"[INFO] Success Inference Im: "<<succInf<<"."<<std::endl;
                        std::cout<<"[INFO] Total Preding BoundingBoxes: "<<infBoxNum<<"."<<std::endl;
                        if (needDetailInfo)
                        {
                            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average Loading Costs: "<<ld / succInf <<"ms/im."<<std::endl;
                            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average LetterBoxed Costs: "<<let / succInf<<"ms/im."<<std::endl;
                            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average Normalized Costs: "<<nor / succInf<<" ms/im"<<std::endl;
                            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average H2D Costs: "<<h2d / succInf<<" ms/im."<<std::endl;
                            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average Inferencing Costs: "<<inf / succInf<<" ms/im."<<std::endl;
                            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average D2H Costs: "<<d2h / succInf<<" ms/im."<<std::endl;
                            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average Post Decode Costs: "<<nms / succInf<<" ms/im."<<std::endl;
                            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average ScaleToOriginal Costs: "<<sto / succInf<<" ms/im."<<std::endl;
                        }
                        std::cout<<std::endl;
                    
                        std::cout<<std::fixed<<std::setprecision(3)<<"[SUMMARY] Average Preprocessing Costs: "<<pre / succInf <<" ms/im." <<std::endl;
                        std::cout<<std::fixed<<std::setprecision(3)<<"[SUMMARY] Average Host -> Host Inferencing Costs: "<<e2e / succInf<<" ms/im."<<std::endl;
                        std::cout<<std::fixed<<std::setprecision(3)<<"[SUMMARY] Average Postprocessing Costs: "<<pos / succInf<<" ms/im."<<std::endl;
                        std::cout<<std::fixed<<std::setprecision(3)<<"[SUMMARY] Average Full Chain Costs (Without Loading): "<<(pre+e2e+pos) / succInf<<" ms/im."<<std::endl
                        <<std::endl;
                    }
                    
                    return true;
                }
        };
    }


    // End2End 类 (输出形如: [batchSize, 300, 6])
    class YOLOEnd2end::Instance : public YOLOBase
    {
        public:
            Instance(int mode, const std::string& modelpath, int deviceid,
                int batchsize,  int imwidth, int imheight, int anchors_num, float confth, int maxdets, int letterboxpadvalue, 
                bool needRecorTime, bool needdetailinfo, std::string modelname, std::optional<std::vector<std::string>> catsname=std::nullopt):
            YOLOBase(mode, modelpath, deviceid, batchsize, imwidth, imheight, confth, maxdets, 
                letterboxpadvalue, needRecorTime, needdetailinfo, modelname, catsname)
            {
                clsNum = 1; scoreNum = 1; anchorsNum = anchors_num;
            }
            ~Instance(){}

            void PostDecode(const float* ptr, std::vector<YOLO::BoundingBox>& outBoxes) override
            {
                for (std::size_t anchor_index=0; anchor_index<anchorsNum; ++anchor_index)
                {
                    int validCount=0;
                    int anchorOffeset = anchor_index * (boxCordsNum + clsNum + scoreNum);
                    float score = *(ptr + anchorOffeset + 4);
                    if (confTh > 0)
                    {
                        if (score < confTh)
                        {
                            continue;
                        }
                    }
                    int cls = static_cast<int>(*(ptr + anchorOffeset + 5));
                    
                    float x1 = *(ptr + anchorOffeset);
                    float y1 = *(ptr + anchorOffeset + 1);
                    float x2 = *(ptr + anchorOffeset + 2);
                    float y2 = *(ptr + anchorOffeset + 3);
    

                    if (!(std::isfinite(x1) && std::isfinite(y1) && std::isfinite(x2) && std::isfinite(y2)) || (x2-x1) <= 0.0f || (y2-y1) <= 0.0f)
                    {
                        continue;
                    }
                    
                    YOLO::BoundingBox bbox;
                    bbox.x1 = x1;
                    bbox.y1 = y1;
                    bbox.x2 = x2;
                    bbox.y2 = y2;
                    bbox.score = score;
                    bbox.cls = cls;
                    outBoxes.push_back(bbox);
                    ++validCount;
                    if (validCount > maxDets) break;
                }
            }
            
            bool checkNmsOpStatus()
            {
                return isNmsop;
            }
    };
    
    YOLOEnd2end::YOLOEnd2end(int mode, const std::string& modelpath, int deviceid, std::string modelname, std::optional<std::vector<std::string>> catsname,
        int batchsize, int imwidth, int imheight, float confth, int maxdets, int anchors_num, int letterboxpadvalue, bool needRecorTime,
        bool needdetailinfo)
    {
        inst_ = std::make_unique<Instance>(mode, modelpath, deviceid, batchsize, imwidth, imheight, anchors_num, confth, 
            maxdets, letterboxpadvalue, needRecorTime, needdetailinfo, modelname, catsname);
    }
    YOLOEnd2end::~YOLOEnd2end(){std::cout<<"[INFO] Model {"+inst_->modelName+"_"<<inst_->modelId<<"} Unload."<<std::endl;}

    bool YOLOEnd2end::Init()
    {
        std::cout<<"[INFO] Model {"+inst_->modelName+"_"<<inst_->modelId<<"} Starts Initializing."<<std::endl;
        auto timeIni0 = std::chrono::steady_clock::now();

        if (inst_->checkNmsOpStatus())
        {
            std::cerr<<"[ERROR] mode 取值范围为 [0, 1, 2]; 原因: End2End 类型的 YOLO 模型内置 NMS 逻辑，额外指定 NMS 算子执行是无效的！"<<std::endl;
            return false;
        }

        if (!inst_->Init())
        {
            std::cerr<<"[ERROR] Model Initializing Failed! "<<std::endl;
            return false;
        }

        auto timeIni1 = std::chrono::steady_clock::now();
        float timeIni = std::chrono::duration<float, std::milli>(timeIni1-timeIni0).count();
        std::cout<<"[INFO] Model Initializing Costed "<<std::fixed<<std::setprecision(3)<<timeIni<<" ms."<<std::endl;
        return true;
    }

    bool YOLOEnd2end::Execute(const std::vector<std::filesystem::path>& imFilesPathVec, std::vector<YOLO::Detection>& outPreds, 
        std::optional<std::string> saveDir, bool verbose)
    {
        return inst_ -> Execute(imFilesPathVec, outPreds, verbose, saveDir);
    }


    // AnchorFree 类 (输出形如: [batchSize, boxcordsnum+clsnum, anchorsnum])
    class YOLOAnchorFree::Instance : public YOLOBase
    {
        public:
            Instance(int mode, const std::string& modelpath, int deviceid, int batchsize,  int imwidth, int imheight, int classesnum,
                float confth, float iouth, int maxdets, int letterboxpadvalue, bool needRecorTime, bool needdetailinfo, std::string modelname,
                std::optional<std::vector<std::string>> catsname=std::nullopt):
            YOLOBase(mode, modelpath, deviceid, batchsize, imwidth, imheight, confth, maxdets, 
                letterboxpadvalue, needRecorTime, needdetailinfo, modelname, catsname)
            {
                clsNum = classesnum; scoreNum = 0; iouTh=iouth;
                int small_w = imwidth / 8;
                int small_h = imheight / 8;

                int medium_w = imwidth / 16;
                int medium_h = imheight / 16;

                int large_w = imwidth / 32;
                int large_h = imheight / 32;

                anchorsNum = small_w*small_h + medium_w*medium_h + large_w*large_h;
            }
            ~Instance(){}

            void PostDecode(const float* ptr, std::vector<YOLO::BoundingBox>& outBoxes) override
            {
                std::vector<YOLO::BoundingBox> bboxes;
                bboxes.reserve(anchorsNum);
                for (std::size_t anchor_index=0; anchor_index<anchorsNum; ++anchor_index)
                {
                    float score = 0.0f;
                    int cls = -1;
                    for (int cls_idx=0; cls_idx<clsNum; ++cls_idx)
                    {
                        size_t scoreIdx = anchorsNum*(boxCordsNum+cls_idx) + anchor_index;
                        if (score < *(ptr + scoreIdx))
                        {
                            score = *(ptr + scoreIdx);
                            cls = cls_idx;
                        }
                    }
                    if (score < confTh)
                    {
                        continue;
                    }
                    float cx = *(ptr + anchor_index);
                    float cy = *(ptr + anchorsNum*1 + anchor_index);
                    float w = *(ptr + anchorsNum*2 + anchor_index);
                    float h = *(ptr + anchorsNum*3 + anchor_index);

                    if (!(std::isfinite(cx) && std::isfinite(cy) && std::isfinite(w) && std::isfinite(h)) || w <= 0.0f || h <= 0.0f)
                    {
                        continue;
                    }

                    float x1 = cx - w * 0.5f;
                    float y1 = cy - h * 0.5f;
                    float x2 = cx + w * 0.5f;
                    float y2 = cy + h * 0.5f;

                    YOLO::BoundingBox box{x1, y1, x2, y2, score, cls};
                    bboxes.push_back(std::move(box));
                }

                std::stable_sort(bboxes.begin(), bboxes.end(), SortByScore);
                outBoxes.reserve(maxDets);
                while (!bboxes.empty() && outBoxes.size() < maxDets)
                {
                    const YOLO::BoundingBox best = bboxes.front();
                    outBoxes.push_back(std::move(best));
                    bboxes.erase(bboxes.begin());
                    for (int box_idx=0; box_idx<bboxes.size();)
                    {
                        if (best.cls == bboxes[box_idx].cls)
                        {
                            float iou = ComputeIoU(best, bboxes[box_idx]);
                            if (iou > iouTh)
                            {
                                bboxes.erase(bboxes.begin() + static_cast<long long>(box_idx));
                            }
                            else
                            {
                                ++box_idx;
                            }
                        }
                        else
                        {
                            ++box_idx;
                        }
                    }
                }
            }
    };
    
    YOLOAnchorFree::YOLOAnchorFree(int mode, const std::string& modelpath, int deviceid, std::string modelname, std::optional<std::vector<std::string>> catsname, 
    int batchsize, int imwidth, int imheight, int classesnum, float confth, float iouth, int maxdets, int letterboxpadvalue,
    bool needRecorTime, bool needdetailinfo)
    {
        inst_ = std::make_unique<Instance>(mode, modelpath, deviceid, batchsize, imwidth, imheight, classesnum, confth, iouth, maxdets, letterboxpadvalue,
            needRecorTime, needdetailinfo, modelname, catsname);
    }
    YOLOAnchorFree::~YOLOAnchorFree(){std::cout<<"[INFO] Model {"+inst_->modelName+"_"<<inst_->modelId<<"} Unload."<<std::endl;}

    bool YOLOAnchorFree::Init()
    {
        std::cout<<"[INFO] Model {"+inst_->modelName+"_"<<inst_->modelId<<"} Starts Initializing."<<std::endl;
        auto timeIni0 = std::chrono::steady_clock::now();

        if (!inst_->Init())
        {
            std::cerr<<"[ERROR] Model Initializing Failed! "<<std::endl;
            return false;
        }

        auto timeIni1 = std::chrono::steady_clock::now();
        float timeIni = std::chrono::duration<float, std::milli>(timeIni1-timeIni0).count();
        std::cout<<"[INFO] Model Initializing Costed "<<std::fixed<<std::setprecision(3)<<timeIni<<" ms."<<std::endl;
        return true;
    }

    bool YOLOAnchorFree::Execute(const std::vector<std::filesystem::path>& imFilesPathVec, std::vector<YOLO::Detection>& outPreds, 
                std::optional<std::string> saveDir, bool verbose)
    {
        return inst_->Execute(imFilesPathVec, outPreds, verbose, saveDir);
    }


    // AnchorBase 类 (输出形如：[batchSize, anchorsnum, boxcordsnum+1+clsnum])
    class YOLOAnchorBase::Instance : public YOLOBase
    {
        private:
            int small_w, small_h, medium_w, medium_h, large_w, large_h;

            void GetGridStride(int anchorIdx, int& gridX, int& gridY, int& stride)
            {
                if (anchorIdx < (small_w*small_h))
                {
                    gridX = anchorIdx % small_w;
                    gridY = anchorIdx / small_h;
                    stride = 8;
                }
                else if (anchorIdx < (small_w*small_h + medium_w*medium_h))
                {
                    anchorIdx -= (small_w*small_h);
                    gridX = anchorIdx % medium_w;
                    gridY = anchorIdx / medium_h;
                    stride = 16;
                }
                else
                {
                    anchorIdx -= (small_w*small_h + medium_w*medium_h);
                    gridX = anchorIdx % large_w;
                    gridY = anchorIdx / large_h;
                    stride = 32;
                }
            }
        
        public:
            bool isYOLOX{false};

            Instance(int mode, const std::string& modelpath, int deviceid, int batchsize, int imwidth, int imheight, int classesnum,
                float confth, float iouth, int anchorsfactor, int maxdets, int letterboxpadvalue, bool isyolox, bool needrecortime,
                bool needdetailinfo, std::string modelname, std::optional<std::vector<std::string>> catsname=std::nullopt):
                YOLOBase(mode, modelpath, deviceid, batchsize, imwidth, imheight, confth, maxdets, letterboxpadvalue, needrecortime, 
                    needdetailinfo, modelname, catsname)
                {
                    clsNum=classesnum; scoreNum=1; iouTh=iouth; isYOLOX=isyolox;
                    small_w = imwidth / 8;
                    small_h = imheight / 8;

                    medium_w = imwidth / 16;
                    medium_h = imheight / 16;

                    large_w = imwidth / 32;
                    large_h = imheight / 32;
                    anchorsNum = anchorsfactor*(small_w*small_h + medium_w*medium_h + large_w*large_h);
                }
            ~Instance(){}

            void PostDecode(const float* ptr, std::vector<YOLO::BoundingBox>& outBoxes) override
            {
                std::vector<YOLO::BoundingBox> bboxes;
                bboxes.reserve(anchorsNum);
                int stride = boxCordsNum + scoreNum + clsNum;
                for (int anchorIdx=0; anchorIdx<anchorsNum; ++anchorIdx)
                {   
                    // 根据置信度阈值筛选合格预测 BoundingBox
                    const float objscore = *(ptr+4);
                    if (objscore < confTh) {ptr+=stride; continue;}
                    float score = 0;
                    int cls = -1;
                    for (int i=0; i<clsNum; ++i)
                    {
                        float clsScore = *(ptr + 5 + i);
                        if (clsScore < confTh) continue;
                        float curscore = clsScore * objscore;
                        if (curscore > score)
                        {
                            score = curscore;
                            cls = i;
                        }
                    }
                    if (score < confTh) {ptr+=stride; continue;}

                    // 计算 BoundingBox 的左上角与右下角坐标
                    float cx, cy, w, h;
                    if (!isYOLOX)
                    {
                        cx = *(ptr); cy = *(ptr+1); w = *(ptr+2); h = *(ptr+3);

                    }
                    else
                    {
                        int gridx, gridy, stride;
                        GetGridStride(anchorIdx, gridx, gridy, stride);
                        cx = ( *(ptr) + static_cast<float>(gridx)) * static_cast<float>(stride);
                        cy = ( *(ptr+1) + static_cast<float>(gridy)) * static_cast<float>(stride);
                        w = std::exp( *(ptr+2) ) * static_cast<float>(stride);
                        h = std::exp( *(ptr+3) ) * static_cast<float>(stride);
                    }

                    if (!(std::isfinite(cx) && std::isfinite(cy) && std::isfinite(w) && std::isfinite(h)) || w <= 0.0f || h <= 0.0f)
                    {
                        continue;
                    }

                    float x1 = cx - w*0.5f;
                    float y1 = cy - h*0.5f;
                    float x2 = cx + w * 0.5f;
                    float y2 = cy + h * 0.5f;

                    YOLO::BoundingBox box{x1, y1, x2, y2, score, cls};
                    bboxes.push_back(box);

                    ptr += stride;
                }

                std::stable_sort(bboxes.begin(), bboxes.end(), SortByScore);
                outBoxes.reserve(maxDets);
                // 根据 iou 阈值剔除交并比大于阈值的 BoundingBox
                while (!bboxes.empty() && outBoxes.size() < maxDets)
                {
                    const YOLO::BoundingBox best = bboxes.front();
                    outBoxes.push_back(best);
                    bboxes.erase(bboxes.begin());
                    for (int i=0; i<bboxes.size();)
                    {
                        if (best.cls == bboxes[i].cls)
                        {
                            float iou = ComputeIoU(best, bboxes[i]);
                            if (iou > iouTh)
                            {
                                bboxes.erase(bboxes.begin() + static_cast<long long>(i));
                            }
                            else ++i;
                        }
                        else ++i;
                    }
                }
            }
    }; 
    
    YOLOAnchorBase::YOLOAnchorBase(int mode, const std::string& modelpath, int deviceid, std::string modelname, std::optional<std::vector<std::string>> catsname,
                int batchsize, int imwidth, int imheight, int classesnum, float confth, float iouth, int anchorsfactor,
                int maxdets, int letterboxpadvalue, bool isyolox, bool needRecorTime, bool needdetailinfo)
    {
        inst_=std::make_unique<Instance>(mode, modelpath, deviceid, batchsize, imwidth, imheight, classesnum, confth, iouth, anchorsfactor, maxdets,
            letterboxpadvalue, isyolox, needRecorTime, needdetailinfo, modelname, catsname);
    }
    YOLOAnchorBase::~YOLOAnchorBase(){std::cout<<"[INFO] Model {"+inst_->modelName+"_"<<inst_->modelId<<"} Unload."<<std::endl;}

    bool YOLOAnchorBase::Init()
    {
        std::cout<<"[INFO] Model {"+inst_->modelName+"_"<<inst_->modelId<<"} Starts Initializing."<<std::endl;
        auto timeIni0 = std::chrono::steady_clock::now();

        if (!inst_->Init())
        {
            std::cerr<<"[ERROR] Model Initializing Failed! "<<std::endl;
            return false;
        }

        auto timeIni1 = std::chrono::steady_clock::now();
        float timeIni = std::chrono::duration<float, std::milli>(timeIni1-timeIni0).count();
        std::cout<<"[INFO] Model Initializing Costed "<<std::fixed<<std::setprecision(3)<<timeIni<<" ms."<<std::endl;
        return true;
    }

    bool YOLOAnchorBase::Execute(const std::vector<std::filesystem::path>& imFilesPathVec, std::vector<YOLO::Detection>& outPreds, 
                std::optional<std::string> saveDir, bool verbose)
    {
        return inst_->Execute(imFilesPathVec, outPreds, verbose, saveDir, inst_->isYOLOX);
    }
}

