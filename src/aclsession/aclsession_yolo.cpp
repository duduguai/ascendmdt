#include <iostream>
#include <chrono>
#include "aclsession_yolo.h"

namespace AI::CV::YOLO
{
    namespace
    {
        /// @brief DVPP 内存对齐宽、高计算工具
        /// @param value 需要对齐的值
        /// @param factor 对齐因子
        /// @return 对齐后的值
        uint32_t AlignUp(uint32_t value, uint32_t factor)
        {
            return (value + factor - 1) / factor * factor;
        }
    }

    bool AclSessionYOLO::CreateDvppChannel(acldvppChannelDesc*& desc, acldvppChannelMode mode, std::string modename)
    {
        desc = acldvppCreateChannelDesc();

        ret = acldvppSetChannelDescMode(desc, mode);
        if (!check("Set ChannelDesc Mode: "+modename, true)) return false;

        ret = acldvppCreateChannel(desc);
        if (!check("Create Channel: "+modename, true)) return false;

        return true;
    }

    bool AclSessionYOLO::CheckTensorConsist(int requireNum, int mode)
    {
        bool ok{false}; int actualNum{-1};

        ok = std::visit([&](auto& exec)->bool
        {
            if (mode == 0)
            {
                using ExecT = std::decay_t<decltype(exec)>;

                if constexpr (!std::is_same_v<typename ExecT::InType, DvppIn>)
                {
                    actualNum = exec.dataIn.data->size();
                    if (actualNum != requireNum) return false;
                }
                return true;
            }
            else
            {
                actualNum = exec.dataOut.data->size();
                if (actualNum != requireNum) return false;
            }
            return true;
        }, *paramPtr);

        if (!ok)
        {
            std::cout<<"[ERROR] "+ioType[mode]+", SessionExecParam 与模型要求的 Tensor 数量不同!\n"
                     <<"        SessionExecParam 中 Tensor 数量: "<<actualNum<<",\n"
                     <<"        模型要求输入的 Tensor 数量:       "<<requireNum<<"."<<std::endl;
            return false;
        }

        return true;
    }

    bool AclSessionYOLO::CheckBytesConsist(int bufferSz, int mode, int idx)
    {
        bool ok{false}; int actualBytes{-1};

        ok = std::visit([&](auto& exec)->bool
        {
            if (mode == 0)
            {
                using ExecT = std::decay_t<decltype(exec)>;

                if constexpr (!std::is_same_v<typename ExecT::InType, DvppIn>)
                {
                    if constexpr (std::is_same_v<typename ExecT::InType, BaseIn>)
                    {
                        actualBytes = exec.dataIn.data->at(idx).size() * sizeof(float);
                    }
                    else
                    {
                        actualBytes = exec.dataIn.data->at(idx).size();
                    }
                    if (actualBytes != bufferSz) return false;
                }
            }
            else
            {
                actualBytes = exec.dataOut.data->at(idx).size()*sizeof(float);
                if (actualBytes != bufferSz) return false;
            }
            return true;
        }, *paramPtr);

        if (!ok)
        {
            std::cout<<"[ERROR] "+ioType[mode]+", SessionExecParam 中 Tensor 数据量与模型要求不同! \n"
                     <<"        Tensor 序号:         "<<idx<<", \n"
                     <<"        Param Tensor 数据量: "<<actualBytes<<" Bytes, \n"
                     <<"        模型要求数据量:      "<<bufferSz<<" Bytes."<<std::endl;
            return false;
        }
        return true;
    }

    bool AclSessionYOLO::PrepareBuffer(int idx, void*& buffer, int bufferSz, int mode)
    {
        return std::visit([&](auto& exec)->bool 
        {
            using ExecT = std::decay_t<decltype(exec)>;

            if constexpr (std::is_same_v<typename ExecT::InType, DvppIn>)
            {
                if (mode == 0)
                {
                    ret = acldvppMalloc(&buffer, bufferSz);
                    return check("Malloc for Input via DVPP", true);
                }
            }

            if (runMode == ACL_DEVICE)
            {
                if (mode == 0) buffer = exec.dataIn.data->at(idx).data();
                else           buffer = exec.dataOut.data->at(idx).data();
                return true;
            }

            ret = aclrtMalloc(&buffer, bufferSz, ACL_MEM_MALLOC_HUGE_FIRST);
            return check("Malloc for "+ioType[mode], true);
        }, *paramPtr);
    }

    bool AclSessionYOLO::Host2DeviceEP()
    {
        if (runMode == ACL_HOST)
        {
            return std::visit([&](auto& exec)->bool
            {
                using ExecT = std::decay_t<decltype(exec)>;

                if constexpr (!std::is_same_v<typename ExecT::InType, DvppIn>)
                {
                    for (int i=0; i<inputDataInfos.size(); ++i)
                    {
                        ret = aclrtMemcpy(inputDataInfos[i].buffer, inputDataInfos[i].size, exec.dataIn.data->at(i).data(),
                                    inputDataInfos[i].size, ACL_MEMCPY_HOST_TO_DEVICE);
                        
                        if (!check("Host Input Data Transfer to Device", true)) return false;
                    }
                }

                return true;
            }, *paramPtr);
        }
        return true;
    }

    bool AclSessionYOLO::DvppDecode()
    {
        return std::visit([&](auto& exec)->bool
        {
            using ExecT = std::decay_t<decltype(exec)>;

            if constexpr (std::is_same_v<typename ExecT::InType, DvppIn>)
            {
                int oriSz, decSz;
                int32_t components;
                uint32_t width, height, widthStride, heightStride;
                acldvppPicDesc* imDecPicDesc;
                std::chrono::steady_clock::time_point timeH2d0, timeH2d1, timeDec0, timeDec1;

                for (int i=0; i<batchSize; ++i)
                {
                    oriSz = exec.dataIn.data->at(i).size();
                    // EP 模式下首先需要将 Host 内存中的数据转移到 Device 内存上
                    if (runMode==ACL_HOST)
                    {
                        if (timeInfoPtr) timeH2d0 = std::chrono::steady_clock::now();

                        if (oriSz > dvpp->defaultBinSz->at(i)) 
                        {
                            std::cout<<"[INFO] 输入图像原始二进制数据大小大于默认 2 MB, 需要在 Device 上重新分配内存空间."<<std::endl;

                            ret = aclrtFree(dvpp->imsBinVecPtr->at(i)); 
                            if (!check("释放为二进制数据默认申请的内存空间", true)) return false;

                            ret = aclrtMalloc(&(dvpp->imsBinVecPtr->at(i)), oriSz, ACL_MEM_MALLOC_HUGE_FIRST);
                            if (!check("为二进制数据重新申请内存空间", true)) return false;
                            dvpp->defaultBinSz->at(i) = oriSz;
                        }

                        ret = aclrtMemcpy(dvpp->imsBinVecPtr->at(i), oriSz, exec.dataIn.data->at(i).data(), oriSz, ACL_MEMCPY_HOST_TO_DEVICE);
                        if (!check("DVPP, Host Data Transfer To Device", true)) return false;

                        if (timeInfoPtr)
                        {
                            timeH2d1 = std::chrono::steady_clock::now();
                            timeInfoPtr->h2d += std::chrono::duration<float, std::milli>(timeH2d1-timeH2d0).count();
                        }

                        /// DVPP Decode 信息加载
                        if (timeInfoExtra) timeDec0 = std::chrono::steady_clock::now();
                        // 获取图像数据的基本信息 (在 CPU 上处理的，不需要使用 Device 上的内存)
                        ret = acldvppJpegGetImageInfo(exec.dataIn.data->at(i).data(), oriSz, &width, &height, &components);
                        if (!check("DVPP, 根据原始二进制数据获取图像基本信息", true)) return false;
                        exec.dataIn.meta->at(i).imOriW = width;
                        exec.dataIn.meta->at(i).imOriH = height;

                        widthStride = AlignUp(width, 64); heightStride = AlignUp(height, 16); // * 此处针对 Atlas 推理产品对 宽stride*高stride 有 64*16 对齐的约束

                        // 判断是否需要重新为 Decode 图像分配内存空间
                        decSz = widthStride*heightStride*3/2; // * 此处计算大小仅针对 YUV_420_SP 格式，后续考虑泛用性更新
                        if (decSz > dvpp->defaultDecSz->at(i)) 
                        {
                            std::cout<<"[INFO] 解码后图像数据所需空间大于默认 4 MB, 需要在 Device 上重新分配内存空间."<<std::endl;

                            ret = acldvppFree(dvpp->imsDecVecPtr->at(i)); 
                            if (!check("释放为解码数据默认申请的内存空间", true)) return false;

                            ret = acldvppMalloc(&(dvpp->imsDecVecPtr->at(i)), decSz);
                            if (!check("为解码数据重新申请内存空间", true)) return false;
                            dvpp->defaultDecSz->at(i) = decSz;
                        }

                        // 为 Decode 图像填充描述符
                        imDecPicDesc = acldvppGetPicDesc(dvpp->srcBtPicDesc, i);
                        acldvppSetPicDescData(imDecPicDesc, dvpp->imsDecVecPtr->at(i));
                        acldvppSetPicDescSize(imDecPicDesc, decSz);
                        acldvppSetPicDescFormat(imDecPicDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
                        acldvppSetPicDescWidth(imDecPicDesc, width);
                        acldvppSetPicDescHeight(imDecPicDesc, height);
                        acldvppSetPicDescWidthStride(imDecPicDesc, widthStride);
                        acldvppSetPicDescHeightStride(imDecPicDesc, heightStride);

                        // 挂起 DVPP 解码任务
                        ret = acldvppJpegDecodeAsync(dvpp->JPEGD, dvpp->imsBinVecPtr->at(i), oriSz, imDecPicDesc, stream);
                        if (!check("DVPP, JPEG 解码任务挂载", true)) return false;

                        if (timeInfoExtra)
                        {
                            timeDec1 = std::chrono::steady_clock::now();
                            timeInfoExtra->ld += std::chrono::duration<float, std::milli>(timeDec1-timeDec0).count();
                        }
                    }

                    if (timeInfoExtra) timeDec0 = std::chrono::steady_clock::now();
                    ret = aclrtSynchronizeStream(stream);
                    if (!check("DVPP, 等待 JPEG 任务解码同步", true)) return false;
                    if (timeInfoExtra)
                    {
                        timeDec1 = std::chrono::steady_clock::now();
                        timeInfoExtra->ld += std::chrono::duration<float, std::milli>(timeDec1-timeDec0).count();
                    }
                }
            }
            
            return true;
        }, *paramPtr);
    }

    bool AclSessionYOLO::DvppLetterBox()
    {
        return std::visit([&](auto& exec)->bool
        {
            using ExecT = std::decay_t<decltype(exec)>;

            if constexpr (std::is_same_v<typename ExecT::InType, DvppIn>)
            {
                acldvppPicDesc* srcPicDesc;
                acldvppPicDesc* dstPicDesc;
                float scaleW, scaleH, scaleFactor;
                uint32_t width, height, rzwidth, rzheight, padL, padT, padR, padB;
                std::chrono::steady_clock::time_point timeLett0, timeLett1;

                if (timeInfoExtra) timeLett0 = std::chrono::steady_clock::now();
                for (int i=0; i<batchSize; ++i)
                {
                    srcPicDesc = acldvppGetPicDesc(dvpp->srcBtPicDesc, i);
                    width = acldvppGetPicDescWidth(srcPicDesc);
                    height = acldvppGetPicDescHeight(srcPicDesc);

                    // 计算 LetterBox 变换参数
                    scaleW = static_cast<float>(dvpp->imWidth) / static_cast<float>(width);
                    scaleH = static_cast<float>(dvpp->imHeight) / static_cast<float>(height);
                    scaleFactor = std::min(scaleW, scaleH);
                    rzwidth = width * scaleFactor; rzheight = height * scaleFactor;
                    exec.dataIn.meta->at(i).scale = scaleFactor;

                    // 注意！ DVPP 中使用 Padding 工具时需要注意 Padding 值需要是偶数，否则在执行过程中会引发内存分配错误
                    padL = (dvpp->imWidth-rzwidth)/2; padL = (padL%2==0) ? padL:padL+1;
                    padT = (dvpp->imHeight-rzheight)/2; padT = (padT%2==0) ? padT:padT+1;
                    padR = dvpp->imWidth - rzwidth - padL; padR = (padR%2==0) ? padR:padR+1;
                    padB = dvpp->imHeight - rzheight - padT; padB = (padB%2==0) ? padB:padB+1;

                    exec.dataIn.meta->at(i).left = padL;
                    exec.dataIn.meta->at(i).top = padT;
                    exec.dataIn.meta->at(i).right = padR;
                    exec.dataIn.meta->at(i).bottom = padB;

                    // 为 cropAreas 赋值，指定 srcPic 中有效的图像区域
                    acldvppSetRoiConfig(dvpp->cropAreasPtr->at(i), 0, width-1, 0, height-1);
                    // 为 borderCfgs 赋值，指定 Pad 的区域
                    acldvppSetBorderConfigLeft(dvpp->borderCfgsPtr->at(i), padL);
                    acldvppSetBorderConfigTop(dvpp->borderCfgsPtr->at(i), padT);
                    acldvppSetBorderConfigRight(dvpp->borderCfgsPtr->at(i), padR);
                    acldvppSetBorderConfigBottom(dvpp->borderCfgsPtr->at(i), padB);
                }

                // DVPP，挂载 Crop + Resize + Padding 任务
                ret = acldvppVpcBatchCropResizeMakeBorderAsync(dvpp->VPC, dvpp->srcBtPicDesc, dvpp->roiNumsPtr->data(), batchSize, dvpp->dstBtPicDesc, 
                            dvpp->cropAreasPtr->data(), dvpp->borderCfgsPtr->data(), dvpp->resizeCfg, stream);
                if (!check("DVPP, 挂载 Crop + Resize + Padding 任务", true)) return false;

                // DVPP, 等待 Crop + Resize + Padding 任务同步
                ret = aclrtSynchronizeStream(stream);
                if (!check("DVPP, 等待 Crop + Resize + Padding 任务同步", true)) return false;

                if (timeInfoExtra)
                {
                    timeLett1 = std::chrono::steady_clock::now();
                    timeInfoExtra->let += std::chrono::duration<float, std::milli>(timeLett1-timeLett0).count();
                }
            }

            return true;
        }, *paramPtr);
    }

    bool AclSessionYOLO::PreProcess()
    {
        std::chrono::steady_clock::time_point timeH2D0, timeH2D1;

        if (timeInfoPtr) timeH2D0=std::chrono::steady_clock::now();
        if (!Host2DeviceEP()) return false;
        if (timeInfoPtr)
        {
            timeH2D1 = std::chrono::steady_clock::now();
            timeInfoPtr->h2d += std::chrono::duration<float, std::milli>(timeH2D1-timeH2D0).count();
        }

        if (timeInfoExtra) timeInfoExtra->Init();

        if (!DvppDecode()) return false;

        return DvppLetterBox();
    }

    bool AclSessionYOLO::NmsOpExecute()
    {
        return std::visit([&](auto& exec)->bool
        {
            using ExecT = std::decay_t<decltype(exec)>;

            if constexpr (std::is_same_v<typename ExecT::OutType, NmsOpOut>)
            {
                std::chrono::steady_clock::time_point timeNms0, timeNms1;

                if (timeInfoExtra) timeNms0 = std::chrono::steady_clock::now();
                aclnnStatus status;
                status = aclnnNonMaxSuppressionGetWorkspaceSize(nms->boxes, nms->scores, nms->maxDetsPerClaIntArr, nms->iouThFloArr, 
                        nms->confThFloArr, 1, nms->indices, &(nms->workSpaceSz), &(nms->executor));
                if (status != 0) 
                {
                    std::cout<<"[ERROR] NMS OP, 确定 NMS OP 执行所需的 Work Space Size Failed! Error Code: "<<status<<std::endl;
                    if (status == 361001) std::cout<<"[ERROR] 当前设备型号不支持该算子执行！"<<std::endl;
                    return false;
                }

                status = aclnnNonMaxSuppression(nms->workSpaceAddr, nms->workSpaceSz, nms->executor, stream);
                if (status != 0) 
                {
                    std::cout<<"[ERROR] NMS OP, 挂载 NMS 算子执行任务 Failed! Error Code: "<<status<<std::endl;
                    if (status == 361001) std::cout<<"[ERROR] 当前设备型号不支持该算子执行！"<<std::endl;
                    return false;
                }

                ret = aclrtSynchronizeStream(stream);
                if (!check("NMS OP, 等待 NMS 算子执行任务同步", true)) return false;

                if (timeInfoExtra)
                {
                    timeNms1 = std::chrono::steady_clock::now();
                    timeInfoExtra->nms += std::chrono::duration<float, std::milli>(timeNms1-timeNms0).count();
                }
            }
            
            return true;
        }, *paramPtr);
    }

    bool AclSessionYOLO::Device2HostEP()
    {
        if (runMode == ACL_HOST)
        {
            return std::visit([&](auto& exec)->bool 
            {
                using ExecT = std::decay_t<decltype(exec)>;

                if constexpr (!std::is_same_v<typename ExecT::OutType, NmsOpOut>)
                {
                    for (int i=0; i<outputDataInfos.size(); ++i)
                    {
                        ret = aclrtMemcpy(exec.dataOut.data->at(i).data(), outputDataInfos[i].size, outputDataInfos[i].buffer,
                                    outputDataInfos[i].size, ACL_MEMCPY_DEVICE_TO_HOST);
                        if (!check("Device Output Data Transfer to Host", true)) return false;
                    }
                }
                else
                {
                    ret = aclrtMemcpyAsync(exec.dataOut.data->at(0).data(), outputDataInfos[0].size, outputDataInfos[0].buffer,
                                outputDataInfos[0].size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
                    if (!check("NMS OP, EP 模式下挂载 box 数据 D2H 任务", true)) return false;

                    ret = aclrtMemcpyAsync(exec.dataOut.data->at(1).data(), outputDataInfos[1].size, outputDataInfos[1].buffer,
                                outputDataInfos[1].size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
                    if (!check("NMS OP, EP 模式下挂载 score 数据 D2H 任务", true)) return false;

                    /// 测试用后续移除
                    if (nms->indicesDevSz == 0 || !nms->indicesAddr)
                    {
                        std::cout<<"[DEBUG] YOLO Session NMS 算子专用参数 [nmsopCoeffi, nmsopCoeffi]未成功初始化!"<<std::endl;
                        return false;
                    }

                    ret = aclrtMemcpyAsync(exec.dataOut.indices->data(), nms->indicesDevSz, nms->indicesAddr, nms->indicesDevSz, 
                                ACL_MEMCPY_DEVICE_TO_HOST, stream);
                    if (!check("NMS OP, EP 模式下挂载 indices 数据 D2H 任务", true)) return false;

                    ret = aclrtSynchronizeStream(stream);
                    if (!check("NMS OP, EP 模式下等待模型输出数据 D2H 传输同步", true)) return false;
                }

                return true;
            }, *paramPtr);
        }

        return true;
    }

    bool AclSessionYOLO::PostProcess()
    {
        std::chrono::steady_clock::time_point timeD2H0, timeD2H1;

        if (!NmsOpExecute()) return false;

        if (timeInfoPtr) timeD2H0 = std::chrono::steady_clock::now();
        if (!Device2HostEP()) return false;
        if (timeInfoPtr)
        {
            timeD2H1 = std::chrono::steady_clock::now();
            timeInfoPtr->d2h += std::chrono::duration<float, std::milli>(timeD2H1-timeD2H0).count();
        }

        return true;
    }


    //// PUBLIC 方法

    AclSessionYOLO::AclSessionYOLO(const std::string& modelpath, int deviceid, YOLOParam* paramptr, int batchsize, bool needtimeinfo): 
            AclSession(modelpath, deviceid, batchsize, needtimeinfo), paramPtr(paramptr)
    {
        std::visit([&](auto& exec)
        {
            using ExecT = std::decay_t<decltype(exec)>;

            if constexpr (std::is_same_v<typename ExecT::InType, DvppIn> || std::is_same_v<typename ExecT::OutType, NmsOpOut>)
            {
                if (needtimeinfo) timeInfoExtra = std::make_unique<TimeInfoSessionExtra>();
            }

        }, *paramPtr);
    }

    AclSessionYOLO::~AclSessionYOLO()
    {
        std::visit([&](auto& exec)
        {
            using ExecT = std::decay_t<decltype(exec)>;   
            if constexpr (std::is_same_v<typename ExecT::InType, DvppIn>)
            {   
                // 销毁 LetterBox 中的 Roiconfig 和 BorderConfig 以及 ResizeConfig
                if (dvpp->resizeCfg) acldvppDestroyResizeConfig(dvpp->resizeCfg);
                if (dvpp->cropAreasPtr)
                {
                    for (int i=0; i<batchSize; ++i) {if (dvpp->cropAreasPtr->at(i)) acldvppDestroyRoiConfig(dvpp->cropAreasPtr->at(i));}
                }
                if (dvpp->borderCfgsPtr)
                {
                    for (int i=0; i<batchSize; ++i) {if (dvpp->borderCfgsPtr->at(i)) acldvppDestroyBorderConfig(dvpp->borderCfgsPtr->at(i));}
                }
                // 销毁 LetterBox 输入和输出图像 Batch 描述符
                if (dvpp->srcBtPicDesc) acldvppDestroyBatchPicDesc(dvpp->srcBtPicDesc);
                if (dvpp->dstBtPicDesc) acldvppDestroyBatchPicDesc(dvpp->dstBtPicDesc);

                // 销毁为 Binary 数据和 Decode 数据创建的内存空间
                if (dvpp->imsDecVecPtr)
                {
                    for (int i=0; i<batchSize; ++i){if (dvpp->imsDecVecPtr->at(i)) acldvppFree(dvpp->imsDecVecPtr->at(i));}
                }
                if (dvpp->imsBinVecPtr)
                {
                    for (int i=0; i<batchSize; ++i){if (dvpp->imsBinVecPtr->at(i)) aclrtFree(dvpp->imsBinVecPtr->at(i));}
                }

                // 销毁通道描述符
                if (dvpp->VPC) {acldvppDestroyChannel(dvpp->VPC); acldvppDestroyChannelDesc(dvpp->VPC);}
                if (dvpp->JPEGD) {acldvppDestroyChannel(dvpp->JPEGD); acldvppDestroyChannelDesc(dvpp->JPEGD);}
            }
            
            if constexpr (std::is_same_v<typename ExecT::OutType, NmsOpOut>)
            {
                // 释放 WorkSpace 内存空间
                if (nms->workSpaceAddr) aclrtFree(nms->workSpaceAddr);

                // 销毁 IntArray 与 FloatArray
                if (nms->maxDetsPerClaIntArr) aclDestroyIntArray(nms->maxDetsPerClaIntArr);
                if (nms->confThFloArr) aclDestroyFloatArray(nms->confThFloArr);
                if (nms->iouThFloArr) aclDestroyFloatArray(nms->iouThFloArr);

                // 销毁 Tensor
                if (nms->boxes) aclDestroyTensor(nms->boxes);
                if (nms->scores) aclDestroyTensor(nms->scores);
                if (nms->indices) aclDestroyTensor(nms->indices);

                // EP模式，销毁为 Indices 在 Device 上的内存空间
                if (runMode == ACL_HOST)
                {
                    if (nms->indicesAddr) aclrtFree(nms->indicesAddr);
                }
            }
        }, *paramPtr);
    }

    bool AclSessionYOLO::Init()
    {
        if (!BaseInit()) return false;

        return std::visit([&](auto& exec)->bool
        {
            using ExecT = std::decay_t<decltype(exec)>;

            // 使用 DVPP 时需要额外针对 DVPP 需要使用的资源进行初始化
            if constexpr (std::is_same_v<typename ExecT::InType, DvppIn>)
            {   
                // 初始化 DVPP 使用的参数
                dvpp = std::make_unique<DvppCoeffi>();
                dvpp->imWidth = exec.dataIn.imwidth;
                dvpp->imHeight = exec.dataIn.imheight;
                dvpp->lettPad = exec.dataIn.lettpad;

                // 计算模型输入图像宽、高经过对齐后的值 
                int imWidthStride = AlignUp(dvpp->imWidth, 64); // * 64 是针对于 Atlas 推理系列产品 (310B / 310P), 后续需要进一步更新泛用性
                int imHeightStride = AlignUp(dvpp->imHeight, 16); // * 16 是针对于 Atlas 推理系列产品 (310B / 310P), 后续需要进一步更新泛用性
                int bytesPerIm = (dvpp->imWidth) * (dvpp->imHeight)*3/2; // * 单位为 Bytes, 此处计算的是经 DVPP Decode 和 LetterBox 后单张图像的大小，格式采用的是 YUV_420_SP, 后续需要进一步更新泛用性

                // 创建 DVPP 操作所需要的 Channel 
                if (!CreateDvppChannel(dvpp->JPEGD, DVPP_CHNMODE_JPEGD, "JPEGD")) return false;
                if (!CreateDvppChannel(dvpp->VPC, DVPP_CHNMODE_VPC, "VPC")) return false;

                // 创建保存原始 Binary 数据的地址变量
                dvpp->defaultBinSz = std::make_unique<std::vector<int>>(batchSize, 2*1024*1024);
                dvpp->imsBinVecPtr = std::make_unique<std::vector<void*>>(batchSize);
                for (int i=0; i<batchSize; ++i)
                {
                    // EP 模式下，需要在 Device 端创建保存 Binary 数据的内存空间
                    if (runMode==ACL_HOST)
                    {
                        ret = aclrtMalloc(&(dvpp->imsBinVecPtr->at(i)), dvpp->defaultBinSz->at(i), ACL_MEM_MALLOC_HUGE_FIRST);
                        if (!check("使用 DVPP, 在 Device 上为原始二进制数据创建内存空间", true)) return false;
                    }
                    // RC 模式下，仅需要将 param 中 DVPP 下输入参数的首地址绑定到 vector 中
                    else
                    {
                        dvpp->imsBinVecPtr->at(i) = exec.dataIn.data->at(i).data();
                        // imsBinVecPtr->at(i) = (*paramPtr).dvppParam.dataInDvpp->at(i).data();
                    }
                }

                // 为 DVPP 解码后的中间数据创建内存空间
                dvpp->defaultDecSz = std::make_unique<std::vector<int>>(batchSize, 4*1024*1024);
                dvpp->imsDecVecPtr = std::make_unique<std::vector<void*>>(batchSize);
                for (int i=0; i<batchSize; ++i)
                {
                    ret = acldvppMalloc(&(dvpp->imsDecVecPtr->at(i)), dvpp->defaultDecSz->at(i));
                    if (!check("使用 DVPP, 为 Decode 中间数据创建内存空间", true)) return false;
                }

                // 为 DVPP 实现 LetterBox 构造所需的 输入(src) 和 输出(dst) 信息描述符
                dvpp->srcBtPicDesc = acldvppCreateBatchPicDesc(batchSize);
                dvpp->dstBtPicDesc = acldvppCreateBatchPicDesc(batchSize);
                acldvppPicDesc* dstPicDesc;
                void* dstBufferPtr = inputDataInfos[0].buffer; // 当前 DVPP 仅支持模型输入中当且仅当第一个参数为输入图像数据的情况，后续可能考虑进一步的泛用性更新
                for (int i=0; i<batchSize; ++i)
                {
                    dstPicDesc = acldvppGetPicDesc(dvpp->dstBtPicDesc, i);
                    acldvppSetPicDescData(dstPicDesc, dstBufferPtr);
                    acldvppSetPicDescSize(dstPicDesc, bytesPerIm);
                    acldvppSetPicDescFormat(dstPicDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
                    acldvppSetPicDescWidth(dstPicDesc, dvpp->imWidth);
                    acldvppSetPicDescHeight(dstPicDesc, dvpp->imHeight);
                    acldvppSetPicDescWidthStride(dstPicDesc, imWidthStride);
                    acldvppSetPicDescHeightStride(dstPicDesc, imHeightStride);

                    char* dstCharPtr = (char*)dstBufferPtr + bytesPerIm;
                    dstBufferPtr = dstCharPtr;
                }

                // 为 DVPP 实现 LetterBox 构造所需的操作参数描述
                dvpp->roiNumsPtr = std::make_unique<std::vector<uint32_t>>(batchSize, 1);
                dvpp->cropAreasPtr = std::make_unique<std::vector<acldvppRoiConfig*>>(batchSize);
                dvpp->borderCfgsPtr = std::make_unique<std::vector<acldvppBorderConfig*>>(batchSize);
                for (int i=0; i<batchSize; ++i)
                {
                    dvpp->cropAreasPtr->at(i) = acldvppCreateRoiConfig(0, 0, 0, 0);
                    dvpp->borderCfgsPtr->at(i) = acldvppCreateBorderConfig();
                    acldvppSetBorderConfigBorderType(dvpp->borderCfgsPtr->at(i), BORDER_CONSTANT);
                    acldvppSetBorderConfigValue(dvpp->borderCfgsPtr->at(i), 0, dvpp->lettPad);
                    acldvppSetBorderConfigValue(dvpp->borderCfgsPtr->at(i), 1, 128);
                    acldvppSetBorderConfigValue(dvpp->borderCfgsPtr->at(i), 2, 128);
                }
                dvpp->resizeCfg = acldvppCreateResizeConfig();
            }


            // 使用 NMS 算子时需要额外针对 NMS 算子需要使用的资源进行初始化
            if constexpr (std::is_same_v<typename ExecT::OutType, NmsOpOut>)
            {   
                // 初始化 NMS 算子需要的参数
                nms = std::make_unique<NmsOpCoeffi>();
                nms->anchorsNum = exec.dataOut.anchorsnum;
                nms->clsNum = exec.dataOut.clsnum;
                nms->maxDetsPerCla = static_cast<int64_t>(exec.dataOut.maxdetspercla);
                nms->confTh = exec.dataOut.confth;
                nms->iouTh = exec.dataOut.iouth;

                // 确定输出 Tensor 的形状
                std::vector<int64_t> boxShape{static_cast<int64_t>(batchSize), static_cast<int64_t>(nms->anchorsNum), 4};
                std::vector<int64_t> boxStride{static_cast<int64_t>(4* (nms->anchorsNum)), 4, 1};
                std::vector<int64_t> scoreShape{static_cast<int64_t>(batchSize), static_cast<int64_t>(nms->clsNum), 
                                                static_cast<int64_t>(nms->anchorsNum)};
                std::vector<int64_t> scoreStride{static_cast<int64_t>((nms->anchorsNum) * (nms->clsNum)), 
                                                static_cast<int64_t>(nms->anchorsNum), 1};
                std::vector<int64_t> indicesShape{static_cast<int64_t>((batchSize*nms->maxDetsPerCla) * (nms->clsNum)), 3};
                std::vector<int64_t> indicesStride{3, 1};

                // 对于 EP 模式，需要在 Device 上为 NMS OP 输出 Indices 创建内存空间，对于 RC 模式，直接将 param 中的输出与其绑定
                if (runMode == ACL_HOST)
                {
                    nms->indicesDevSz = batchSize * (nms->maxDetsPerCla) * (nms->clsNum) * 3*sizeof(int32_t);
                    ret = aclrtMalloc(&(nms->indicesAddr), nms->indicesDevSz, ACL_MEM_MALLOC_HUGE_FIRST);
                    if (!check("NMS OP, 为算子输出的筛选 Indices 在 Device 上创建内存空间", true)) return false;
                }
                else
                {
                    nms->indicesAddr = exec.dataOut.indices->data();
                }

                // 构建 NMS OP 需要的输入和输出 Tensor 
                nms->boxes = aclCreateTensor(boxShape.data(), boxShape.size(), aclDataType::ACL_FLOAT, boxStride.data(), 0, 
                        aclFormat::ACL_FORMAT_ND, boxShape.data(), boxShape.size(), outputDataInfos[0].buffer);
                nms->scores = aclCreateTensor(scoreShape.data(), scoreShape.size(), aclDataType::ACL_FLOAT, scoreStride.data(), 0, 
                        aclFormat::ACL_FORMAT_ND, scoreShape.data(), scoreShape.size(), outputDataInfos[1].buffer);
                nms->indices = aclCreateTensor(indicesShape.data(), indicesShape.size(), aclDataType::ACL_INT32, indicesStride.data(), 0, 
                        aclFormat::ACL_FORMAT_ND, indicesShape.data(), indicesShape.size(), (nms->indicesAddr));

                // 构建 NMS OP 执行所需的 Array 常量
                nms->maxDetsPerClaIntArr = aclCreateIntArray(&(nms->maxDetsPerCla), 1);
                nms->confThFloArr = aclCreateFloatArray(&(nms->confTh), 1);
                nms->iouThFloArr = aclCreateFloatArray(&(nms->iouTh), 1);

                // 计算 workSpaceSize 并完成 workSpace 内存空间分配
                aclnnStatus status;
                status = aclnnNonMaxSuppressionGetWorkspaceSize(nms->boxes, nms->scores, nms->maxDetsPerClaIntArr,
                    nms->iouThFloArr, nms->confThFloArr, 1, nms->indices, &(nms->workSpaceSz), 
                    &(nms->executor));
                if (status != 0) 
                {
                    std::cout<<"[ERROR] NMS OP, 确定 NMS OP 执行所需的 Work Space Size Failed! Error Code: "<<status<<std::endl;
                    if (status == 361001) std::cout<<"[ERROR] 当前设备型号不支持该算子执行！"<<std::endl;
                    return false;
                }
                ret = aclrtMalloc(&(nms->workSpaceAddr), (nms->workSpaceSz), ACL_MEM_MALLOC_HUGE_FIRST);
                if (!check("NMS OP, 为 NMS OP 算子执行创建工作所需的内存空间", true)) return false;
            }

            return true;
        }, *paramPtr);
    }

    void AclSessionYOLO::DvppApdateAddr(void* addr, int index)
    {
        if (runMode == ACL_DEVICE)
        {
            if (index >= batchSize)
            {
                throw std::runtime_error("[ERROR] DVPP, 输入地址索引更新的序号超出 Batch 范围.");
            }

            dvpp->imsBinVecPtr->at(index) = addr;
        }
    }

    TimeInfoSessionExtra AclSessionYOLO::GetTimeInfoExtra()
    {
        if (!timeInfoExtra) throw std::runtime_error("[ERROR] TimeInfoExtra 未成功初始化.");

        return *timeInfoExtra;
    }
}