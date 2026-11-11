#include "models_core_yolo.h"
#include "tools_cv.h"
#include "tools_base.h"

namespace AI::CV::YOLO
{
    namespace
    {
        /// @brief 制图的颜色取值空间
        const std::vector<int> colors = {235, 219, 11, 243, 243, 243, 183, 223, 0, 221, 111, 255, 0, 237, 204, 68, 243, 0, 255, 255, 0, 179, 255, 1,
        11, 255, 162, 255, 42, 4, 79, 68, 255, 255, 0, 189, 255, 180, 0, 186, 0, 221, 0, 192, 38, 255, 36, 125, 104, 0, 123, 108, 
        27, 255, 47, 109, 252, 104, 31, 17};

        /// @brief 根据模式初始化YOLO Model中的Param参数
        /// @param mode 模式 {0: Base, 1: Aipp, 2: Dvpp, 3: NmsOP, 4: Aipp+NmsOP, 5: Dvpp+NmsOP}
        /// @return Param参数 YOLOParam, 类型为 std::variant
        YOLOParam InitParamByRunTime(int mode)
        {
            switch (mode)
            {
            case 0: return BaseParam{};
            case 1: return AippParam{};
            case 2: return DvppParam{};
            case 3: return BaseNmsParam{};
            case 4: return AippNmsParam{};
            case 5: return DvppNmsParam{};
            
            default: throw std::invalid_argument("\n[ERROR] 无效 mode. 取值空间 {0: Normal, 1: AIPP, 2: DVPP+AIPP, 3: NMSOP, 4: AIPP+NMSOP, 5: DVPP+AIPP+NmsOP}.");
            }
        }

        /// @brief 按照置信度为 BoundingBox 排序
        /// @param a BoundingBox
        /// @param b BoundingBox
        /// @return a 的 score 是否大于 b 的 score
        bool SortByScore(const BoundingBox & a, const BoundingBox & b)
        {
            return a.score > b.score;
        }

        /// @brief 计算两个 BoundingBox 的交并比
        /// @param box1 BoundingBox
        /// @param box2 BoundingBox
        /// @return box1 与 box2 的交并比
        float ComputeIoU(const BoundingBox& box1, const BoundingBox& box2)
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
    }

    //// BaseCore
    // Protected
    void BaseCore::PostDecode(const float* ptr, std::vector<BoundingBox>& outBoxes)
    {
        std::cout<<"[ERROR] 基类不提供 Det 模型 PostDecode 的实现方法, 需要派生类进行重载."<<std::endl;
    }

    void BaseCore::PostDecode(const float* detPtr, const float* segPtr, std::vector<BoundingBox>& boxes,
                    std::vector<cv::Mat>& masks)
    {
        std::cout<<"[ERROR] 基类不提供 Seg 模型 PostDecode 的实现方法, 需要派生类进行重载."<<std::endl;
    }

    // Private 
    bool BaseCore::CollectFiles(const std::string& input)
    {
        imPaths = CollectImageFiles(input);

        if (imPaths.empty())
        {
            std::cout<<"[ERROR] No images found in: "<<input<<"."<<std::endl;
            return false;
        }
        return true;
    }

    int BaseCore::GetEpoches()
    {
        int epoches, total;
        total = imPaths.size();

        if (total%batchSize == 0) epoches = total/batchSize;
        else                      epoches = total/batchSize + 1;

        return epoches;
    }

    bool BaseCore::LoadBatchImMat(int offset, int& validNum)
    {
        int totalFilenum = imPaths.size();
        int index = offset;
        for (int i=0; i<batchSize;)
        {
            if (index < totalFilenum)
            {
                if (loadImMat(imPaths[index].string(), coeIn->imVecMat[i]))
                {
                    imValidIdx[i] = index;
                    ++validNum;
                    ++i;
                }
                ++index;
            }
            else
            {
                if (i==0) return false;
                coeIn->imVecMat[i] = coeIn->imVecMat[i-1];
                imValidIdx[i] = -1;
                ++i;
            }
        }
        return true;
    }

    bool BaseCore::LoadBatchImBin(std::vector<std::vector<uint8_t>>& imBinBatch, std::vector<int>& recordSz, int offset, int& validNum)
    {
        int totalFilenum = imPaths.size();
        int index = offset;
        for (int i=0; i<batchSize;)
        {
            if (index < totalFilenum)
            {
                size_t size;
                if (loadImBinary(imPaths[index].string(), imBinBatch[i], size))
                {
                    if (size > recordSz[i]) {session->DvppApdateAddr(imBinBatch[i].data(), i); recordSz[i]=size;}
                    imValidIdx[i] = index;
                    ++validNum;
                    ++i;
                }
                ++index;
            }
            else
            {
                if (i==0) return false;
                imBinBatch[i] = imBinBatch[i-1];
                if (imBinBatch[i].size() > recordSz[i]) {session->DvppApdateAddr(imBinBatch[i].data(), i); recordSz[i]=imBinBatch[i].size();}
                imValidIdx[i] = -1;
                ++i;
            }
        }
        return true;
    }

    void BaseCore::LetterBoxViaOpenCV(const cv::Mat& imIn, cv::Mat& imOut, LetterBoxMeta& meta)
    {
        meta.imOriW = imIn.cols;
        meta.imOriH = imIn.rows;
        const float scaleW = static_cast<float>(coeIn->imWidth) / static_cast<float>(imIn.cols);
        const float scaleH = static_cast<float>(coeIn->imHeight) / static_cast<float>(imIn.rows);
        meta.scale = std::min(scaleW, scaleH);

        const int resizedW = static_cast<int>(std::round(static_cast<float>(imIn.cols)*meta.scale));
        const int resizedH = static_cast<int>(std::round(static_cast<float>(imIn.rows)*meta.scale));

        meta.left = (coeIn->imWidth - resizedW + 1)/2;
        meta.top = (coeIn->imHeight - resizedH + 1)/2;
        meta.right = coeIn->imWidth - resizedW - meta.left;
        meta.bottom = coeIn->imHeight - resizedH - meta.top;

        cv::Mat imResized;
        cv::resize(imIn, imResized, cv::Size(resizedW, resizedH), 0, 0, cv::INTER_LINEAR);

        cv::copyMakeBorder(
            imResized,
            imOut,
            meta.top,
            meta.bottom,
            meta.left,
            meta.right,
            cv::BORDER_CONSTANT,
            cv::Scalar(coeIn->lettpad, coeIn->lettpad, coeIn->lettpad)
        );
    }

    void BaseCore::Normalize(const cv::Mat& imIn, std::vector<float>& dataOut, int idx)
    {
        float inv;
        if (isYOLOX) inv = 1.0f;
        else         inv = 1/255.0f;
        
        float* r = dataOut.data() + idx * coeIn->numsPerIm + 0 * coeIn->numsPerChannel;
        float* g = dataOut.data() + idx * coeIn->numsPerIm + 1 * coeIn->numsPerChannel;
        float* b = dataOut.data() + idx * coeIn->numsPerIm + 2 * coeIn->numsPerChannel;
        
        for (int i=0; i<imIn.rows; ++i)
        {
            const uint8_t* row = imIn.ptr<uint8_t>(i);
            for (int j=0; j<imIn.cols; ++j)
            {
                int index = i*imIn.cols + j;
                int rowIndex = 3*j;
                r[index] = static_cast<float>(row[rowIndex + 2]) * inv;
                g[index] = static_cast<float>(row[rowIndex + 1]) * inv;
                b[index] = static_cast<float>(row[rowIndex + 0]) * inv;
            }
        }
    }

    void BaseCore::PostDecode(const float* boxesPtr, const float* scoresPtr, const int32_t* indicesPtr,
                std::vector<int>& boxStride, std::vector<int>& scoreStride,
                std::vector<BoundingBox>& outBoxes, int numsPerData, int idx)
    {
        int32_t btIdx, claIdx, boxIdx, scoreOffset, boxOffset;
        float score, cx, cy, w, h, x1, y1, x2, y2;

        outBoxes.reserve(numsPerData);
        for (int i=0; i<numsPerData; ++i)
        {
            btIdx = *(indicesPtr);
            if (btIdx >= 0)
            {
                if (btIdx != idx) std::cout<<"[DEBUG] 预期 batchIdx: "<<idx<<"实际读取 batchIdx: "<<btIdx<<std::endl;
                claIdx = *(indicesPtr+1);
                boxIdx = *(indicesPtr+2);

                scoreOffset = btIdx*scoreStride[0] + claIdx*scoreStride[1] + boxIdx;
                boxOffset = btIdx*boxStride[0] + boxIdx*boxStride[1];

                score = *(scoresPtr + scoreOffset);
                cx = *(boxesPtr + boxOffset);
                cy = *(boxesPtr + boxOffset + 1);
                w = *(boxesPtr + boxOffset + 2);
                h = *(boxesPtr + boxOffset + 3);

                x1 = cx - w*0.5f;
                y1 = cy - h*0.5f;
                x2 = cx + w*0.5f;
                y2 = cy + h*0.5f;

                BoundingBox box{x1, y1, x2, y2, score, claIdx};
                outBoxes.push_back(std::move(box));
            }

            indicesPtr += 3;
        }
    }

    void BaseCore::BBoxToOri(OutputPerIm& output, const LetterBoxMeta& meta)
    {
        float invScale = meta.scale > 0.0f ? (1.0f / meta.scale) : 1.0f;

        for (std::size_t idx=0; idx<output.det.boxes.size();)
        {
            output.det.boxes[idx].x1 = (output.det.boxes[idx].x1 - static_cast<float>(meta.left)) * invScale;
            output.det.boxes[idx].y1 = (output.det.boxes[idx].y1 - static_cast<float>(meta.top)) * invScale;
            output.det.boxes[idx].x2 = (output.det.boxes[idx].x2 - static_cast<float>(meta.left)) * invScale;
            output.det.boxes[idx].y2 = (output.det.boxes[idx].y2 - static_cast<float>(meta.top)) * invScale;

            output.det.boxes[idx].x1 = std::clamp(output.det.boxes[idx].x1, 0.0f, static_cast<float>(meta.imOriW - 1));
            output.det.boxes[idx].y1 = std::clamp(output.det.boxes[idx].y1, 0.0f, static_cast<float>(meta.imOriH - 1));
            output.det.boxes[idx].x2 = std::clamp(output.det.boxes[idx].x2, 0.0f, static_cast<float>(meta.imOriW - 1));
            output.det.boxes[idx].y2 = std::clamp(output.det.boxes[idx].y2, 0.0f, static_cast<float>(meta.imOriH - 1));
            if (output.det.boxes[idx].x1>=output.det.boxes[idx].x2 || output.det.boxes[idx].y1 >= output.det.boxes[idx].y2)
            {
                output.det.boxes.erase(output.det.boxes.begin() + static_cast<long long>(idx));
                if (coeSeg) output.seg->masks.erase(output.seg->masks.begin() + static_cast<long long>(idx));
            }
            else
            {
                ++idx;
            }
        }
    }

    bool BaseCore::MaskToOri(OutputPerIm& output, const LetterBoxMeta& meta, int imwidth, int imheight)
    {
        if (output.det.boxes.size() != output.seg->masks.size())
        {
            std::cout<<"\n[DEBUG] 模型输出保留的 BoundingBox 框数量与保留的掩码区域 Mask 数量不匹配.\n"
                     <<"          BoundingBox Size: "<<output.det.boxes.size()<<".\n"
                     <<"          Mask Size:        "<<output.seg->masks.size()<<"."<<std::endl;
            return false;
        }

        int left, top, rswidth, rsheight;
        left = meta.left;
        top = meta.top;
        rswidth = imwidth - meta.left - meta.right;
        rsheight = imheight - meta.top - meta.bottom;

        for (int i=0; i<output.det.boxes.size(); ++i)
        {
            cv::Mat maskRs0;
            // Resize 到模型输入的 height x width
            cv::resize(output.seg->masks[i], maskRs0, cv::Size(imwidth, imheight), 0, 0, cv::INTER_LINEAR);

            // 转换为 binary Mask
            cv::Mat maskBin;
            cv::threshold(maskRs0, maskBin, coeSeg->maskTh, 1.0, cv::THRESH_BINARY);
            maskBin.convertTo(maskBin, CV_8U);

            // 裁剪有效区域
            maskBin = maskBin(cv::Rect(left, top, rswidth, rsheight)).clone();

            // Resize 到原始图像大小
            cv::Mat maskRs1;
            cv::resize(maskBin, maskRs1, cv::Size(meta.imOriW, meta.imOriH), 0, 0, cv::INTER_LINEAR);

            int x1 = std::max(0, static_cast<int>(output.det.boxes[i].x1));
            int y1 = std::max(0, static_cast<int>(output.det.boxes[i].y1));
            int x2 = std::min(meta.imOriW, static_cast<int>(output.det.boxes[i].x2));
            int y2 = std::min(meta.imOriH, static_cast<int>(output.det.boxes[i].y2));

            // 只保留框中的掩码区域
            cv::Mat maskCrop = cv::Mat::zeros(maskRs1.size(), maskRs1.type());
            if (x2>x1 && y2>y1)
            {
                cv::Rect roi(x1, y1, x2-x1, y2-y1);
                maskRs1(roi).copyTo(maskCrop(roi));
            }

            output.seg->masks[i] = std::move(maskCrop);
        }

        return true;
    }

    void BaseCore::DrawBoundingBox(cv::Mat& im, const OutputPerIm& output)
    {
        int im_width = im.cols;
        for (int i=0; i<output.det.boxes.size(); ++i)
        {
            int colorIdx = output.det.boxes[i].cls % 20;
            int b = colors[3*colorIdx];
            int g = colors[3*colorIdx+1];
            int r = colors[3*colorIdx+2];

            cv::Scalar boxcolor = cv::Scalar(b, g, r);

            // 将掩码区域在原始图像中着色
            if (output.seg)
            {
                cv::Mat overlay = im.clone();
                overlay.setTo(boxcolor, output.seg->masks[i]);
                cv::addWeighted(overlay, 0.4, im, 0.6, 0.0, im);
            }

            cv::Scalar textcolor;
            if (colorIdx < 9)
            {
                textcolor = cv::Scalar(104, 31, 17);
            }
            else
            {
                textcolor = cv::Scalar(255, 255, 255);
            }

            float p1x = output.det.boxes[i].x1;
            float p1y = output.det.boxes[i].y1;
            float p2x = output.det.boxes[i].x2;
            float p2y = output.det.boxes[i].y2;

            cv::rectangle(
                im,
                cv::Point(p1x, p1y),
                cv::Point(p2x, p2y),
                boxcolor,
                2,
                cv::LINE_AA
            );

            // 处理 BBox 的标注信息
            std::string label;
            std::ostringstream oss;
            if (!catsName.empty()) oss << catsName.at(output.det.boxes[i].cls)<<": "<< std::fixed << std::setprecision(2) << output.det.boxes[i].score;
            else oss << output.det.boxes[i].cls<<": "<< std::fixed << std::setprecision(2) << output.det.boxes[i].score;
            label = oss.str();

            cv::Size text_size = cv::getTextSize(label, 0, 0.6, 2, 0);
            text_size.height += 3;
            bool outside = (p1y >= (text_size.height));
            if (p1x + text_size.width > im_width)
            {
                p1x = im_width - text_size.width;
            }
            p2x = p1x + text_size.width;
            if (outside) p2y = p1y - text_size.height;
            else p2y = p1y + text_size.height;

            cv::rectangle(
                im,
                cv::Point(p1x, p1y),
                cv::Point(p2x, p2y),
                boxcolor,
                -1,
                cv::LINE_AA
            );

            if (outside) p1y = p1y - 2;
            else p1y = p1y + text_size.height - 1;

            cv::putText(
                im,
                label,
                cv::Point(p1x, p1y),
                0,
                0.6,
                textcolor,
                2,
                cv::LINE_AA
            );
        }
    }

    // Public
    BaseCore::BaseCore(int mode, const std::string& modelpath, int deviceid, int batchsize, int imwidth, int imheight, float confTh, 
        bool recordtime, int maxdets, int clsnum, int lettpad, std::optional<std::vector<std::string>> catsname, 
        std::optional<bool> enableseg): 
        batchSize(batchsize)
    {
        if (catsname) catsName = std::move(*catsname);

        imValidIdx.resize(batchsize);

        param = InitParamByRunTime(mode);

        if (recordtime) timeInfo = std::make_unique<TimeInfoYOLO>();

        // 当部署 Seg 模型时
        if (enableseg && *enableseg)
        {
            coeSeg = std::make_unique<SegCoe>();

            coeSeg->protoWidth  = imwidth/4;
            coeSeg->protoHeight = imheight/4;

            coeSeg->numsPerMask = coeSeg->protoWidth * coeSeg->protoHeight;
            coeSeg->numsPerProto = coeSeg->masknum * coeSeg->numsPerMask;
        }
        
        std::visit([&](auto& exec)
        {
            using ExecT = std::decay_t<decltype(exec)>;

            if constexpr (std::is_same_v<typename ExecT::InType, DvppIn>)
            {
                exec.dataIn.imwidth = imwidth;
                exec.dataIn.imheight = imheight;
                exec.dataIn.lettpad = lettpad;
            }
            else
            {
                int numsPerChannel = imwidth * imheight;
                int numsPerIm = 3*numsPerChannel;
                coeIn = std::make_unique<BaseCoeForIn>(BaseCoeForIn{imwidth, imheight, numsPerChannel, numsPerIm, lettpad});
                coeIn->imVecMat.resize(batchsize);
                coeIn->meta.resize(batchsize);
            }

            if constexpr (std::is_same_v<typename ExecT::OutType, NmsOpOut>)
            {
                exec.dataOut.clsnum = clsnum;
                int maxdetspercla = ((maxdets/clsnum) < 15) ? 15:(maxdets/clsnum);
                exec.dataOut.maxdetspercla = maxdetspercla;
                exec.dataOut.confth = confTh;
            }
            else
            {
                coeOut = std::make_unique<BaseCoeForOut>();
                coeOut->clsNum = clsnum;
                coeOut->maxDets = maxdets;
                coeOut->confTh = confTh;
            }
        }, param);

        session = std::make_unique<AclSessionYOLO>(modelpath, deviceid, &param, batchsize, recordtime);

        id = _count;
        ++_count;
    }

    BaseCore::~BaseCore(){}

    bool BaseCore::Init()
    {
        std::visit([&](auto& exec)
        {
            using ExecT = std::decay_t<decltype(exec)>;

            if constexpr (std::is_same_v<typename ExecT::InType, DvppIn>)
            {
                exec.dataIn.data = std::make_unique<std::vector<std::vector<uint8_t>>>(batchSize);
                exec.dataIn.curSz = std::make_unique<std::vector<int>>(batchSize, 2*1024*1024);
                for (int i=0; i<batchSize; ++i) exec.dataIn.data->at(i).reserve(exec.dataIn.curSz->at(i));
                exec.dataIn.meta = std::make_unique<std::vector<LetterBoxMeta>>(batchSize);
            }
            else if constexpr (std::is_same_v<typename ExecT::InType, AippIn>)
            {
                exec.dataIn.data = std::make_unique<std::vector<std::vector<uint8_t>>>(1);
                exec.dataIn.data->at(0).resize(batchSize*coeIn->numsPerIm);
            }
            else
            {
                exec.dataIn.data = std::make_unique<std::vector<std::vector<float>>>(1);
                exec.dataIn.data->at(0).resize(batchSize*coeIn->numsPerIm);
            }


            if constexpr (std::is_same_v<typename ExecT::OutType, NmsOpOut>)
            {
                exec.dataOut.boxStride = {exec.dataOut.anchorsnum*4, 4};
                exec.dataOut.scoreStride = {exec.dataOut.anchorsnum*exec.dataOut.clsnum, exec.dataOut.anchorsnum};

                // 使用 SEG 时, 需要对其 onnx 节点图进行修改, 将输出中 box score mask 拆开分为三部分
                if (coeSeg)  exec.dataOut.data = std::make_unique<std::vector<std::vector<float>>>(4);
                else         exec.dataOut.data = std::make_unique<std::vector<std::vector<float>>>(2);
                
                exec.dataOut.data->at(0).resize(batchSize * exec.dataOut.anchorsnum * boxCordsNum);
                exec.dataOut.data->at(1).resize(batchSize * exec.dataOut.anchorsnum * exec.dataOut.clsnum);
                exec.dataOut.indices = std::make_unique<std::vector<int32_t>>(batchSize * exec.dataOut.maxdetspercla * exec.dataOut.clsnum * 3);

                if (coeSeg) {exec.dataOut.data->at(2).resize(batchSize * exec.dataOut.anchorsnum * coeSeg->masknum);
                             exec.dataOut.data->at(3).resize(batchSize * coeSeg->numsPerProto);}
            }
            else
            {
                if (coeSeg) {
                    exec.dataOut.data = std::make_unique<std::vector<std::vector<float>>>(2);
                    exec.dataOut.data->at(0).resize(batchSize * coeOut->anchorsNum * (boxCordsNum + coeOut->scoreNum + coeOut->clsNum + coeSeg->masknum));
                    exec.dataOut.data->at(1).resize(batchSize * coeSeg->numsPerProto);}
                else {
                    exec.dataOut.data = std::make_unique<std::vector<std::vector<float>>>(1);
                    exec.dataOut.data->at(0).resize(batchSize * coeOut->anchorsNum * (boxCordsNum + coeOut->scoreNum + coeOut->clsNum));}
            }
        }, param);

        return session->Init();
    }

    bool BaseCore::Execute(std::vector<std::filesystem::path> imBatchPaths, std::vector<OutputPerIm>& outputs, bool verbose)
    {
        if (imBatchPaths.empty())
        {
            std::cout<<"[ERROR] 推理输入数据为空, 退出推理流程."<<std::endl;
            return false;
        }

        if (imBatchPaths.size() > batchSize)
        {
            std::cout<<"[ERROR] 单次推理数据量大于 BatchSize, 退出推理流程."<<std::endl;
            return false;
        }

        std::chrono::steady_clock::time_point time0, time1;
        if (verbose) time0 = std::chrono::steady_clock::now();

        // 转移输入的图像文件路径到 Core 内部的维护图像路径的变量中
        imPaths = std::move(imBatchPaths);

        bool succ = std::visit([&](auto& exec)->bool
        {
            using ExecT = std::decay_t<decltype(exec)>;

            // 加载并解码图像文件
            int validnum{0};
            if constexpr (std::is_same_v<typename ExecT::InType, DvppIn>)
            {
                if (!LoadBatchImBin(*(exec.dataIn.data), *(exec.dataIn.curSz), 0, validnum))
                {
                    std::cout<<"[ERROR] 输入数据中的图像全部无法正确读取."<<std::endl;
                    return false;
                }
            }
            else
            {
                if (!LoadBatchImMat(0, validnum))
                {
                    std::cout<<"[ERROR] 输入数据中的图像全部无法完成读取和解码."<<std::endl;
                    return false;
                }
            }

            if (verbose) std::cout<<"[INFO] Model { "<<name<<"_"<<id<<" } Starts Processing "<< batchSize<<" Data."<<std::endl;

            // 输入数据前处理 (非 DVPP)
            if constexpr (!std::is_same_v<typename ExecT::InType, DvppIn>)
            {
                int offset{0};
                for (int i=0; i<batchSize; ++i)
                {
                    cv::Mat imLett;
                    LetterBoxViaOpenCV(coeIn->imVecMat[i], imLett, coeIn->meta[i]);

                    if constexpr (std::is_same_v<typename ExecT::InType, BaseIn>)
                    {
                        Normalize(imLett, exec.dataIn.data->at(0), i);
                    }
                    else
                    {
                        if (!imLett.isContinuous()) imLett = imLett.clone();
                        std::memcpy(exec.dataIn.data->at(0).data()+offset, imLett.data, coeIn->numsPerIm);
                        offset += coeIn->numsPerIm;
                    }
                }
            }

            // 执行 Session 推理
            if (!session->Execute())
            {
                std::cout<<"[ERROR] Session.Execute() 执行失败."<<std::endl;
                return false;
            }

            /// 推理输出后处理
            outputs.reserve(batchSize);
            int numsPerTarOut;
            if constexpr (std::is_same_v<typename ExecT::OutType, BaseOut>)
            {
                if (!coeSeg) numsPerTarOut = (boxCordsNum + coeOut->scoreNum + coeOut->clsNum) * coeOut->anchorsNum;
                else         numsPerTarOut = (boxCordsNum + coeOut->scoreNum + coeOut->clsNum + coeSeg->masknum) * coeOut->anchorsNum;
            }
            else numsPerTarOut =  exec.dataOut.indices->size() / batchSize;
            
            for (int i=0; i<batchSize; ++i)
            {
                if (imValidIdx[i] == -1) continue;
                OutputPerIm output;
                output.filename = imPaths[imValidIdx[i]].filename().string();

                // Decode (NMS)
                if constexpr (std::is_same_v<typename ExecT::OutType, BaseOut>)
                {
                    if (coeSeg)
                    {
                        const float* detPtr = exec.dataOut.data->at(0).data() + i*numsPerTarOut;
                        const float* segPtr = exec.dataOut.data->at(1).data() + i*coeSeg->numsPerProto;
                        output.seg = std::make_unique<Segmantation>();
                        PostDecode(detPtr, segPtr, output.det.boxes, output.seg->masks);
                    }
                    else
                    {
                        const float* ptrBase = exec.dataOut.data->at(0).data() + i*numsPerTarOut;
                        PostDecode(ptrBase, output.det.boxes);
                    }
                }
                else
                {
                    if (coeSeg)
                    {
                        std::cout<<"\n[ERROR] 暂时未支持 Segmantation 模型使用 NMS 算子."<<std::endl;
                        return false;
                    }
                    else
                    {
                        const float* boxesPtr = exec.dataOut.data->at(0).data();
                        const float* scoresPtr = exec.dataOut.data->at(1).data();
                        const int32_t* indicesPtr = exec.dataOut.indices->data() + i*numsPerTarOut;
                        PostDecode(boxesPtr, scoresPtr, indicesPtr, exec.dataOut.boxStride, exec.dataOut.scoreStride, output.det.boxes, numsPerTarOut, i);
                    }
                }

                if constexpr (std::is_same_v<typename ExecT::InType, DvppIn>)
                {
                    BBoxToOri(output, exec.dataIn.meta->at(i));
                    if (coeSeg) MaskToOri(output, exec.dataIn.meta->at(i), exec.dataIn.imwidth, exec.dataIn.imheight);
                }
                else
                {
                    BBoxToOri(output, coeIn->meta[i]);
                    if (coeSeg) MaskToOri(output, coeIn->meta[i], coeIn->imWidth, coeIn->imHeight); 
                }
                outputs.push_back(std::move(output));
            }
            return true;
        }, param);

        if (!succ)
        {
            std::cout<<"[ERROR] 推理执行失败."<<std::endl;
        }

        if (verbose) 
        {
            time1 = std::chrono::steady_clock::now();
            float time = std::chrono::duration<float, std::milli>(time1-time0).count();
            std::cout<<std::fixed<<std::setprecision(2)<<"[INFO] Model { "<<name<<"_"<<id<<" } Inference "<<batchSize<<" Data Successfully! Total Costs "
                     <<time<<" ms. "<<time/batchSize<<" ms/im. "<<batchSize*1000/time<<" fps."<<std::endl;
        }
        return true;
    }

    bool BaseCore::Execute(const std::string& input, std::optional<std::string> saveDir, bool verbose)
    {
        // 处理输入来源, 解析为图像文件路径向量
        if (!CollectFiles(input)) 
        {
            std::cout<<"[ERROR] 推理图像集解析为空, 无法进行推理!"<<std::endl;
            return false;
        }
        // 获取推理完成需要的 Epoch 总数
        int Epoches = GetEpoches();
        // 判断是否需要保存描框结果，若需要保存则判断当前保存目录是否创建
        if (saveDir && *saveDir != "")
        {
            std::filesystem::create_directories(*saveDir);
        }

        std::chrono::steady_clock::time_point timeLd0, timeLd1, timeLet0, timeLet1, timeNor0, timeNor1, timePos0, timePos1, timePos2;
        float pre{0.0f}, e2e{0.0f}, pos{0.0f};
        float ld{0.0f}, let{0.0f}, nor{0.0f}, h2d{0.0f}, inf{0.0f}, d2h{0.0f}, nms{0.0f}, sto{0.0f};

        int succInf{0}, infBoxNum{0};
        std::cout<<"[INFO] Model { "<<name<<"_"<<id<<" } Starts Inferencing."<<std::endl;

        bool succ = std::visit([&](auto& exec)->bool
        {
            using ExecT = std::decay_t<decltype(exec)>;

            for (int epoch=0; epoch<Epoches; ++epoch)
            {
                if (timeInfo) timeInfo->Init();

                ShowProgress(epoch, Epoches, 80);

                int validnum{0};
                
                /// 通过图像路径加载图像数据
                if (timeInfo) timeLd0 = std::chrono::steady_clock::now();
                // BASE 和 AIPP 对应的图像加载路径
                if constexpr (!std::is_same_v<typename ExecT::InType, DvppIn>)
                {
                    if (!LoadBatchImMat(epoch*batchSize, validnum))
                    {
                        std::cout<<"[WARNING] 已无法从图像集中加载足够的图像构成一个Batch. 退出推理循环."<<std::endl;
                        break;
                    }
                }
                else
                {
                    if (!LoadBatchImBin(*(exec.dataIn.data), *(exec.dataIn.curSz), epoch*batchSize, validnum))
                    {
                        std::cout<<"[WARNING] 已无法从图像集中加载足够的图像构成一个Batch. 退出推理循环."<<std::endl;
                        break;
                    }
                }
                if (timeInfo) {timeLd1 = std::chrono::steady_clock::now();
                               timeInfo->ld += std::chrono::duration<float, std::milli>(timeLd1-timeLd0).count();}
                
                /// 图像数据前处理
                if constexpr (!std::is_same_v<typename ExecT::InType, DvppIn>)
                {
                    int offset{0};
                    for (int idx=0; idx<batchSize; ++idx)
                    {
                        cv::Mat imLett;
                        if (timeInfo && imValidIdx[idx] != -1) timeLet0 = std::chrono::steady_clock::now();
                        LetterBoxViaOpenCV(coeIn->imVecMat[idx], imLett, coeIn->meta[idx]);
                        if (timeInfo && imValidIdx[idx] != -1) {timeLet1 = std::chrono::steady_clock::now();
                                                                timeInfo->let += std::chrono::duration<float, std::milli>(timeLet1-timeLet0).count();}
                            
                        if constexpr (std::is_same_v<typename ExecT::InType, BaseIn>)
                        {
                            
                            if (timeInfo && imValidIdx[idx] != -1) timeNor0 = std::chrono::steady_clock::now();
                            Normalize(imLett, exec.dataIn.data->at(0), idx);
                            if (timeInfo && imValidIdx[idx] != -1) {timeNor1 = std::chrono::steady_clock::now();
                                                                    timeInfo->nor += std::chrono::duration<float, std::milli>(timeNor1-timeNor0).count();}
                        }
                        else
                        {
                            if (!imLett.isContinuous()) imLett = imLett.clone();
                            std::memcpy(exec.dataIn.data->at(0).data()+offset, imLett.data, coeIn->numsPerIm);
                            offset += coeIn->numsPerIm;
                        }
                    }
                }
                
                /// 会话推理
                if (!session->Execute())
                {
                    std::cout<<"[WARNING] Epoch "<<epoch<<" 推理失败，跳过该 Batch."<<std::endl;
                    continue;
                }

                /// 推理输出后处理
                int numsPerData;
                if constexpr (!std::is_same_v<typename ExecT::OutType, NmsOpOut>)
                {
                    if (!coeSeg) numsPerData = (boxCordsNum + coeOut->scoreNum + coeOut->clsNum) * coeOut->anchorsNum;
                    else         numsPerData = (boxCordsNum + coeOut->scoreNum + coeOut->clsNum + coeSeg->masknum) * coeOut->anchorsNum;
                }
                else
                {
                    numsPerData = exec.dataOut.indices->size() / batchSize;
                }

                for (int idx=0; idx<batchSize; ++idx)
                {
                    if (imValidIdx[idx] == -1) continue;

                    OutputPerIm output;
                    output.filename = imPaths[imValidIdx[idx]].filename().string();

                    if (timeInfo) timePos0 = std::chrono::steady_clock::now();

                    // Decode (NMS)
                    if constexpr (std::is_same_v<typename ExecT::OutType, BaseOut>)
                    {
                        if (coeSeg)
                        {
                            const float* detPtr = exec.dataOut.data->at(0).data() + idx*numsPerData;
                            const float* segPtr = exec.dataOut.data->at(1).data() + idx*coeSeg->numsPerProto;
                            output.seg = std::make_unique<Segmantation>();
                            PostDecode(detPtr, segPtr, output.det.boxes, output.seg->masks);
                        }
                        else
                        {
                            const float* ptrBase = exec.dataOut.data->at(0).data() + idx*numsPerData;
                            PostDecode(ptrBase, output.det.boxes);
                        }
                    }
                    else
                    {
                        if (coeSeg)
                        {
                            std::cout<<"\n[ERROR] 暂时未支持 Segmantation 模型使用 NMS 算子."<<std::endl;
                            return false;
                        }
                        else
                        {
                            const float* boxesPtr = exec.dataOut.data->at(0).data();
                            const float* scoresPtr = exec.dataOut.data->at(1).data();
                            const int32_t* indicesPtr = exec.dataOut.indices->data() + idx*numsPerData;
                            PostDecode(boxesPtr, scoresPtr, indicesPtr, exec.dataOut.boxStride, exec.dataOut.scoreStride, output.det.boxes, numsPerData, idx);
                        }
                    }
                
                    if (timeInfo) timePos1 = std::chrono::steady_clock::now();

                    if constexpr (std::is_same_v<typename ExecT::InType, DvppIn>)
                    {
                        BBoxToOri(output, exec.dataIn.meta->at(idx));
                        if (coeSeg) MaskToOri(output, exec.dataIn.meta->at(idx), exec.dataIn.imwidth, exec.dataIn.imheight);
                    }
                    else
                    {
                        BBoxToOri(output, coeIn->meta[idx]);
                        if (coeSeg) MaskToOri(output, coeIn->meta[idx], coeIn->imWidth, coeIn->imHeight); 
                    }
                    
                    if (timeInfo) {timePos2 = std::chrono::steady_clock::now();
                                   timeInfo->nms += std::chrono::duration<float, std::milli>(timePos1-timePos0).count();
                                   timeInfo->sto += std::chrono::duration<float, std::milli>(timePos2-timePos1).count();}
                    
                    infBoxNum += output.det.boxes.size();

                    // 描框并保存图像
                    if (saveDir && *saveDir != "")
                    {
                        cv::Mat im;
                        if constexpr (std::is_same_v<typename ExecT::InType, DvppIn>)
                        {
                            im = cv::imread(imPaths[imValidIdx[idx]].string());
                        }
                        else
                        {
                            im = coeIn->imVecMat[idx];
                        }
                        DrawBoundingBox(im, output);
                        std::string savePath = *saveDir + "/" + output.filename;
                        cv::imwrite(savePath, im);
                    }
                }

                if (timeInfo)
                {
                    TimeInfoSessionCore timeInfoCore = session->GetTimeInfo();
                    TimeInfoSessionExtra timeInfoExtra;

                    ld += timeInfo->ld;
                    let += timeInfo->let;
                    nor += timeInfo->nor;
                    if constexpr (std::is_same_v<typename ExecT::InType, DvppIn>)
                    {
                        timeInfoExtra = session->GetTimeInfoExtra();
                        ld += timeInfoExtra.ld;
                        let += timeInfoExtra.let; 
                    }

                    h2d += timeInfoCore.h2d;
                    inf += timeInfoCore.inf;
                    d2h += timeInfoCore.d2h;

                    nms += timeInfo->nms;
                    sto += timeInfo->sto;
                    if constexpr (std::is_same_v<typename ExecT::OutType, NmsOpOut>)
                    {
                        timeInfoExtra = session->GetTimeInfoExtra();
                        nms += timeInfoExtra.nms;
                    }
                }
                // 成功推理的图片计数
                succInf += validnum;
            }
            ShowProgress(Epoches, Epoches, 80);
            std::cout<<"\n"<<std::endl;
            
            return true;
        }, param);

        if (!succ) 
        {
            std::cout<<"[ERROR] 推理执行失败!"<<std::endl;
            return false;
        }

        if (succInf == 0)
        {
            std::cout<<"[ERROR] No images had been inferenced successfully!"<<std::endl;
            return false;
        }

        if (timeInfo)
        {
            pre = let + nor;
            e2e = h2d + inf + d2h;
            pos = nms + sto;
        }

        if (verbose)
        {
            std::cout<<"[INFO] Success Inference Im: "<<succInf<<"."<<std::endl;
            std::cout<<"[INFO] Total Preding BoundingBoxes: "<<infBoxNum<<"."<<std::endl;

            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average LetterBoxed Costs: "<<let / succInf<<"ms/im."<<std::endl;
            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average Normalized Costs: "<<nor / succInf<<" ms/im"<<std::endl;
            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average H2D Costs: "<<h2d / succInf<<" ms/im."<<std::endl;
            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average Inferencing Costs: "<<inf / succInf<<" ms/im."<<std::endl;
            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average D2H Costs: "<<d2h / succInf<<" ms/im."<<std::endl;
            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average Post Decode Costs: "<<nms / succInf<<" ms/im."<<std::endl;
            std::cout<<std::fixed<<std::setprecision(3)<<"[INFO] Average ScaleToOriginal Costs: "<<sto / succInf<<" ms/im."<<std::endl;

            std::cout<<std::endl;
            std::cout<<std::fixed<<std::setprecision(3)<<"[SUMMARY] Average Loading Costs: "<<ld / succInf <<"ms/im."<<std::endl;
            std::cout<<std::fixed<<std::setprecision(3)<<"[SUMMARY] Average Preprocessing Costs: "<<pre / succInf <<" ms/im." <<std::endl;
            std::cout<<std::fixed<<std::setprecision(3)<<"[SUMMARY] Average Host -> Host Inferencing Costs: "<<e2e / succInf<<" ms/im."<<std::endl;
            std::cout<<std::fixed<<std::setprecision(3)<<"[SUMMARY] Average Postprocessing Costs: "<<pos / succInf<<" ms/im."<<std::endl;
            std::cout<<std::fixed<<std::setprecision(3)<<"[SUMMARY] Average Full Chain Costs (Without Loading): "<<(pre+e2e+pos) / succInf<<" ms/im."<<std::endl
            <<std::endl;
        }

        return true;
    }



    //// AnchorBasedCore
    void AnchorBasedCore::GetGridStride(int anchorIdx, int& gridx, int& gridy, int& stride)
    {
        if (anchorIdx < (gridsnumWS*gridsnumHS))
        {
            gridx = anchorIdx % gridsnumWS;
            gridy = anchorIdx / gridsnumHS;
            stride = 8;
        }
        else if (anchorIdx < (gridsnumWS*gridsnumHS + gridsnumWM*gridsnumHM))
        {
            anchorIdx -= (gridsnumWS*gridsnumHS);
            gridx = anchorIdx % gridsnumWM;
            gridy = anchorIdx / gridsnumHM;
            stride = 16;
        }
        else
        {
            anchorIdx -= (gridsnumWS*gridsnumHS + gridsnumWM*gridsnumHM);
            gridx = anchorIdx % gridsnumWL;
            gridy = anchorIdx / gridsnumHL;
            stride = 32;
        }
    }

    void AnchorBasedCore::PostDecode(const float* ptr, std::vector<BoundingBox>& outBoxes)
    {
        const int   N       = coeOut->anchorsNum;
        const int   C       = coeOut->clsNum;
        const int   MaxDets = coeOut->maxDets;
        const float confth  = coeOut->confTh;
        const float iouth   = coeOut->iouTh;
        
        std::vector<BoundingBox> bboxes;
        bboxes.reserve(coeOut->anchorsNum);
        int stride = boxCordsNum + coeOut->scoreNum + C;
        for (int anchorIdx=0; anchorIdx<N; ++anchorIdx)
        {   
            // 根据置信度阈值筛选合格预测 BoundingBox
            const float objscore = ptr[4];
            if (objscore < confth) {ptr+=stride; continue;}
            float score = 0;
            int cls = -1;
            for (int i=0; i < C; ++i)
            {
                float clsScore = ptr[5+i];
                if (clsScore < confth) continue;
                float curscore = clsScore * objscore;
                if (curscore > score)
                {
                    score = curscore;
                    cls = i;
                }
            }
            if (score < confth) {ptr+=stride; continue;}

            // 计算 BoundingBox 的左上角与右下角坐标
            float cx, cy, w, h;
            if (!isYOLOX)
            {
                cx = ptr[0]; cy = ptr[1]; w = ptr[2]; h = ptr[3];

            }
            else
            {
                int gridx, gridy, stride;
                GetGridStride(anchorIdx, gridx, gridy, stride);
                cx = ( ptr[0] + static_cast<float>(gridx)) * static_cast<float>(stride);
                cy = ( ptr[1] + static_cast<float>(gridy)) * static_cast<float>(stride);
                w = std::exp( ptr[2] ) * static_cast<float>(stride);
                h = std::exp( ptr[3] ) * static_cast<float>(stride);
            }

            if (!(std::isfinite(cx) && std::isfinite(cy) && std::isfinite(w) && std::isfinite(h)) || w <= 0.0f || h <= 0.0f)
            {
                ptr+=stride; continue;
            }

            bboxes.push_back({cx - 0.5f*w, cy - 0.5f*h,
                              cx + 0.5f*w, cy + 0.5f*h,
                              score, cls});

            ptr += stride;
        }

        std::stable_sort(bboxes.begin(), bboxes.end(), SortByScore);
        outBoxes.reserve(MaxDets);
        // 根据 iou 阈值剔除交并比大于阈值的 BoundingBox
        while (!bboxes.empty() && outBoxes.size() < MaxDets)
        {
            const BoundingBox best = bboxes.front();
            outBoxes.push_back(best);
            bboxes.erase(bboxes.begin());
            for (int boxIdx=0; boxIdx<bboxes.size();)
            {
                if (best.cls == bboxes[boxIdx].cls)
                {
                    float iou = ComputeIoU(best, bboxes[boxIdx]);
                    if (iou > iouth) bboxes.erase(bboxes.begin() + static_cast<long long>(boxIdx));
                    else             ++boxIdx;
                }
                else ++boxIdx;
            }
        }
    }

    AnchorBasedCore::AnchorBasedCore(int mode, const std::string& modelpath, std::string modelname, int deviceid, int batchsize, int imwidth, int imheight,
        float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad, int anchorsfactor, 
        std::optional<std::vector<std::string>> catsname, bool isYolox): 
        BaseCore(mode, modelpath, deviceid, batchsize, imwidth, imheight, confTh, recordtime, maxdets, clsnum, lettpad, catsname)
    {
        name = modelname;
        isYOLOX = isYolox;

        gridsnumWS = imwidth / 8;
        gridsnumHS = imheight / 8;

        gridsnumWM = imwidth / 16;
        gridsnumHM = imheight / 16;

        gridsnumWL = imwidth / 32;
        gridsnumHL = imheight / 32;

        std::visit([&](auto& exec)
        {
            using ExecT = std::decay_t<decltype(exec)>;
            
            if constexpr (std::is_same_v<typename ExecT::OutType, BaseOut>)
            {
                coeOut->anchorsNum = anchorsfactor*(gridsnumWS*gridsnumHS + gridsnumWM*gridsnumHM + gridsnumWL*gridsnumHL);
                coeOut->scoreNum = 1;
                coeOut->iouTh = iouTh;
            }
            else
            {
                exec.dataOut.anchorsnum = anchorsfactor*(gridsnumWS*gridsnumHS + gridsnumWM*gridsnumHM + gridsnumWL*gridsnumHL);
                exec.dataOut.iouth = iouTh;
            }
        }, param);
    }

    AnchorBasedCore::~AnchorBasedCore(){}



    //// AnchorFreeCore
    void AnchorFreeCore::PostDecode(const float* ptr, std::vector<BoundingBox>& outBoxes)
    {
        const int   N       = coeOut->anchorsNum;
        const int   C       = coeOut->clsNum;
        const int   MaxDets = coeOut->maxDets;
        const float confth  = coeOut->confTh;
        const float iouth   = coeOut->iouTh;

        constexpr int TILE = 512;

        float  bestScore[TILE];
        int    bestCls[TILE];

        std::vector<BoundingBox> bboxes;
        bboxes.reserve(coeOut->anchorsNum);

        for (int t0=0; t0<N; t0 += TILE)
        {
            const int n = std::min(TILE, N-t0);

            // 初始化
            for (int i=0; i<n; ++i) {bestScore[i]=0.0f; bestCls[i]=-1;}

            // 外层循环变量为输出第2维度, 内存循环变量为输出第3维度(最后维度) -> 内层循环访问的内存空间是连续的, 最大化利用L1缓存
            for (int c=0; c<C; ++c)
            {
                const float* clsScore = ptr + N * (boxCordsNum + c) + t0;
                for (int i=0; i<n; ++i)
                {
                    if (clsScore[i] < confth) continue;

                    if (clsScore[i] > bestScore[i])
                    {
                        bestScore[i] = clsScore[i];
                        bestCls[i] = c;
                    }
                }
            }

            for (int i=0; i<n; ++i)
            {
                if (bestScore[i] < confth) continue;

                const int a = t0 + i;
                float cx = ptr[a];
                float cy = ptr[N + a];
                float w  = ptr[2*N + a];
                float h  = ptr[3*N + a];

                if (!(std::isfinite(cx) && std::isfinite(cy) && std::isfinite(w) && std::isfinite(h)) 
                    || w <= 0.0f || h <= 0.0f) continue;
                
                bboxes.push_back({cx - 0.5f*w, cy - 0.5f*h,
                                  cx + 0.5f*w, cy + 0.5f*h,
                                  bestScore[i], bestCls[i]});
            }
        }

        std::stable_sort(bboxes.begin(), bboxes.end(), SortByScore);
        outBoxes.reserve(MaxDets);

        while (!bboxes.empty() && outBoxes.size() < MaxDets)
        {
            const BoundingBox best = bboxes.front();
            outBoxes.push_back(std::move(best));
            bboxes.erase(bboxes.begin());
            for (int boxIdx=0; boxIdx<bboxes.size();)
            {
                if (best.cls == bboxes[boxIdx].cls)
                {
                    float iou = ComputeIoU(best, bboxes[boxIdx]);
                    if (iou > iouth) bboxes.erase(bboxes.begin() + static_cast<long long>(boxIdx));
                    else                     ++boxIdx;
                }
                else ++boxIdx;
            }
        }
    }

    struct InfoDecForSegv8_11
    {
        int anchorIdx;
        BoundingBox box;

        InfoDecForSegv8_11()=default;
        InfoDecForSegv8_11(int idx, BoundingBox bbox): anchorIdx(idx), box(bbox){}
    };

    bool SortByScoreForSeg(const InfoDecForSegv8_11& a, const InfoDecForSegv8_11& b)
    {
        return a.box.score > b.box.score;
    }

    void AnchorFreeCore::PostDecode(const float* detPtr, const float* segPtr, std::vector<BoundingBox>& boxes,
                            std::vector<cv::Mat>& masks)
    {
        const int N        = coeOut->anchorsNum;
        const int C        = coeOut->clsNum;
        const int MaxDets  = coeOut->maxDets;
        const float confth = coeOut->confTh;
        const float iouth  = coeOut->iouTh;

        constexpr int TILE = 512;

        float bestScore[TILE];
        int bestCls[TILE];
        std::vector<InfoDecForSegv8_11> infos;
        infos.reserve(N);
        for (int t0=0; t0<N; t0 += TILE)
        {
            int n = std::min(TILE, N-t0);
            for (int i=0; i<n; ++i) {bestScore[i]=0.0f; bestCls[i]=-1;}

            for (int c=0; c<C; ++c)
            {
                const float* clsScore = detPtr + (boxCordsNum+c)*N + t0;
                for (int i=0; i<n; ++i)
                {
                    if (clsScore[i] < confth) continue;

                    if (clsScore[i] > bestScore[i])
                    {
                        bestScore[i] = clsScore[i];
                        bestCls[i] = c;
                    }
                }
            }

            for (int i=0; i<n; ++i)
            {
                if (bestScore[i] < confth) continue;

                const int a = t0 + i;
                float cx = detPtr[a];
                float cy = detPtr[a + N];
                float w  = detPtr[a + N*2];
                float h  = detPtr[a + N*3];

                if (!(std::isfinite(cx) && std::isfinite(cy) && std::isfinite(w) && std::isfinite(h)) 
                    || w <= 0.0f || h <= 0.0f) continue;
                
                infos.push_back({a, {cx - 0.5f*w, cy - 0.5f*h,
                                     cx + 0.5f*w, cy + 0.5f*h,
                                     bestScore[i], bestCls[i]}});
            }
        }

        std::stable_sort(infos.begin(), infos.end(), SortByScoreForSeg);
        boxes.reserve(MaxDets);

        const int masknum     = coeSeg->masknum;
        const int numsPerMask = coeSeg->numsPerMask;
        const int protowidth  = coeSeg->protoWidth;
        const int protoheight = coeSeg->protoHeight;

        std::vector<const float*> ProtoAddrs(masknum);
        for (int i=0; i<masknum; ++i) ProtoAddrs[i] = segPtr + i*numsPerMask;

        masks.reserve(MaxDets);

        while(!infos.empty() && boxes.size() < MaxDets)
        {
            InfoDecForSegv8_11 best = infos.front();
            boxes.push_back(best.box);

            OriMask orimask(numsPerMask);
            for (int i=0; i<numsPerMask; ++i)
            {
                for (int j=0; j<masknum; ++j) orimask.data[i] += detPtr[N*(boxCordsNum+C+j) + best.anchorIdx] * ProtoAddrs[j][i];
            }
            masks.push_back(std::move(orimask.DataToMat(protowidth, protoheight)));

            infos.erase(infos.begin());

            for (int i=0; i<infos.size();)
            {
                if (best.box.cls == infos[i].box.cls)
                {
                    float iou = ComputeIoU(best.box, infos[i].box);
                    if (iou > iouth) infos.erase(infos.begin() + static_cast<long long>(i));
                    else             ++i;
                }
                else                 ++i;
            }
        }
    }
    
    AnchorFreeCore::AnchorFreeCore(int mode, const std::string& modelpath, std::string modelname, int deviceid, int batchsize, int imwidth, int imheight,
                float confTh, float iouTh, bool recordtime, int maxdets, int clsnum, int lettpad, std::optional<std::vector<std::string>> catsname,
                std::optional<bool> enabelseg):
                BaseCore(mode, modelpath, deviceid, batchsize, imwidth, imheight, confTh, recordtime, maxdets, clsnum, lettpad, catsname, enabelseg)
    {
        name = modelname;

        int gridsnumWidths = imwidth / 8;
        int gridsnumHeights = imheight / 8;

        int gridsnumWidthm = imwidth / 16;
        int gridsnumHeightm = imheight / 16;

        int gridsnumWidthl = imwidth / 32;
        int gridsnumHeightl = imheight / 32;

        std::visit([&](auto& exec)
        {
            using ExecT = std::decay_t<decltype(exec)>;

            if constexpr (std::is_same_v<typename ExecT::OutType, BaseOut>)
            {
                coeOut->scoreNum = 0;
                coeOut->iouTh = iouTh;
                coeOut->anchorsNum = gridsnumWidths*gridsnumHeights + gridsnumWidthm*gridsnumHeightm + gridsnumWidthl*gridsnumHeightl;
            }
            else
            {
                exec.dataOut.anchorsnum = gridsnumWidths*gridsnumHeights + gridsnumWidthm*gridsnumHeightm + gridsnumWidthl*gridsnumHeightl;
                exec.dataOut.iouth = iouTh;
            }
        }, param);
    }

    AnchorFreeCore::~AnchorFreeCore(){}



    /// EndToEndCore
    void EndToEndCore::PostDecode(const float* ptr, std::vector<BoundingBox>& outBoxes)
    {
        const int N = coeOut->anchorsNum;
        const int C = coeOut->clsNum;
        const int S = coeOut->scoreNum;
        const int MaxDets = coeOut->scoreNum;
        const float confth = coeOut->confTh;

        outBoxes.reserve(MaxDets);
        for (int anchorIdx=0; anchorIdx<N; ++anchorIdx)
        {
            int anchorOffeset = anchorIdx * (boxCordsNum + C + S);
            float score = *(ptr + anchorOffeset + 4);
            if (confth > 0)
            {
                if (score < confth) continue;
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
            
            BoundingBox bbox{x1, y1, x2, y2, score, cls};
            outBoxes.push_back(bbox);
            
            if (outBoxes.size() > MaxDets) break;
        }
    }

    void EndToEndCore::PostDecode(const float* detPtr, const float* segPtr, std::vector<BoundingBox>& boxes,
                                  std::vector<cv::Mat>& masks)
    {
        const int masknum = coeSeg->masknum;
        const int numsPerMask = coeSeg->numsPerMask;
        const int protowidth = coeSeg->protoWidth;
        const int protoheight = coeSeg->protoHeight;
        
        const int N        = coeOut->anchorsNum;
        const int MaxDets  = coeOut->maxDets;
        const float confth = coeOut->confTh;

        std::vector<const float*> ProtoAddrs(masknum);
        for (int i=0; i<masknum; ++i) ProtoAddrs[i] = segPtr + i*numsPerMask;

        boxes.reserve(MaxDets);
        masks.reserve(MaxDets);

        int stride = boxCordsNum + 2 + masknum;
        for (int n=0; n<N; ++n)
        {
            float score = detPtr[4];
            if (confth > 0)
            {
                if (score < confth)
                {
                    detPtr += stride;
                    continue;
                }
            }
            int cls = static_cast<int>(detPtr[5]);

            float x1 = detPtr[0];
            float y1 = detPtr[1];
            float x2 = detPtr[2];
            float y2 = detPtr[3];

            if (!(std::isfinite(x1) && std::isfinite(y1) && std::isfinite(x2) && std::isfinite(y2)) || (x2-x1) <= 0.0f || (y2-y1) <= 0.0f)
            {
                detPtr += stride;
                continue;
            }

            boxes.push_back({x1, y1, x2, y2, score, cls});

            OriMask orimask(numsPerMask);

            for (int i=0; i<numsPerMask; ++i)
            {
                for (int j=0; j<masknum; ++j) orimask.data[i] += detPtr[6+j] * ProtoAddrs[j][i];
            }

            masks.push_back(std::move(orimask.DataToMat(protowidth, protoheight)));

            if (boxes.size() > MaxDets) break;

            detPtr += stride;
        }
    }

    EndToEndCore::EndToEndCore(int mode, const std::string& modelpath, std::string modelname, int deviceid, int batchsize, int imwidth, int imheight,
                float confTh, int anchorsnum, bool recordtime, int maxdets, int lettpad, std::optional<std::vector<std::string>> catsname,
                std::optional<bool> enableseg):
                BaseCore(mode, modelpath, deviceid, batchsize, imwidth, imheight, confTh, recordtime, maxdets, 1, lettpad, catsname, enableseg)
    {
        if (mode > 2)
        {
            throw std::invalid_argument("\n[ERROR] 无效 mode 赋值. 取值空间 {0: Normal, 1: AIPP, 2: DVPP+AIPP}.\n        原因: 端到端类型 YOLO 模型, 内部已包含 NMS 逻辑, 不需要后置 NMS 操作. 且不支持 NMS 算子调用.");
        }

        name = modelname;
        coeOut->scoreNum = 1;
        coeOut->anchorsNum = anchorsnum;
    }

    EndToEndCore::~EndToEndCore(){}
}