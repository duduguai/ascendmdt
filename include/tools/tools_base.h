#pragma once
#include <iostream>
#include <iomanip>

namespace AI
{
    /// @brief 动态显示进度状况
    /// @param current 当前轮次
    /// @param total 总轮次
    /// @param barWidth 显示进度条宽度
    void ShowProgress(int current, int total, int barWidth=50)
    {
        float progress = static_cast<float>(current) / static_cast<float>(total);
        int pos = static_cast<int>(progress*barWidth);

        std::cout<<"\r Inferencing ... [";
        for (int i=0; i<barWidth; ++i)
        {
            if (i<pos) std::cout<<"=";
            else if (i==pos) std::cout<<">";
            else std::cout<<" ";
        }
        std::cout<<"]"<<std::setw(3)<<static_cast<int>(progress*100)<<"%  ("<<current<<"/"<<total<<")";
        std::cout<<std::flush;
    }
}