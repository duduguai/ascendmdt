#pragma once
#include <algorithm>
#include <string>
#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>

namespace AI::CV
{
    /// @brief 判断路径是否为图像路径
    /// @param path 需要判断的路径
    /// @return 是否为图像路径
    bool IsImagePath(std::filesystem::path path)
    {
        if (! path.has_extension()) return false;

        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){
            return static_cast<char>(std::tolower(c));});

        return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
        ext == ".webp" || ext == ".tif" || ext == ".tiff";
    }

    /// @brief 对于给定输入路径, 收集该路径下的图像文件
    /// @param input 输入路径
    /// @return 收集的图像文件集合
    std::vector<std::filesystem::path> CollectImageFiles(std::filesystem::path input)
    {   
        std::vector<std::filesystem::path> files;
        if (std::filesystem::is_regular_file(input) && IsImagePath(input))
        {
            files.push_back(input);
            return files;
        }
        if (!std::filesystem::is_directory(input))
        {
            return files;
        }

        for (const auto& entry: std::filesystem::recursive_directory_iterator(input))
        {
            if (entry.is_regular_file() && IsImagePath(entry.path()))
            {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
        return files;
    }

    /// @brief 加载图像二进制数据
    /// @param imPath 图像文件路径
    /// @param imBin 输出二进制向量
    /// @param size 输出数据的大小
    /// @return 是否加载成功
    bool loadImBinary(const std::string& imPath, std::vector<uint8_t>& imBin, size_t& size)
    {
        // 打开文件
        FILE* fp = fopen(imPath.c_str(), "rb");
        if (!fp)
        {
            std::cerr<<"[WARNING] 无法打开图像文件："<<imPath<<"."<<std::endl;
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
            std::cerr<<"[WARNING] 图像文件读取错误: "<<imPath<<"."<<std::endl;
            return false;
        }

        return true;
    }

    /// @brief 通过OpenCV加载图像数据
    /// @param imPath 图像文件路径
    /// @param imMat 输出 cv::Mat 格式图像数据
    /// @return 是否加载成功
    bool loadImMat(const std::string& imPath, cv::Mat& imMat)
    {
        cv::Mat im = cv::imread(imPath, cv::IMREAD_COLOR);
        if (im.empty()) 
        {   
            std::cerr<<"[WARNING] 图像文件无法正常加载: "<<imPath<<"."<<std::endl;
            return false;
        }
        imMat = std::move(im);

        return true;
    }
}