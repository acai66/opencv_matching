#include <stdlib.h>
#include <math.h>
#include <filesystem>  // C++17 标准库中的文件系统库
namespace fs = std::filesystem;

#include "matcher.h"
#ifdef _WIN32
#include "windows.h"
#else
#include <dlfcn.h>
#endif // _WIN32


// 动态库函数指针
typedef template_matching::Matcher* (*InitMD)(const template_matching::MatcherParam);

cv::Rect box;
bool drawing_box = false;
bool change_template = false;

// 鼠标回调函数，用于绘制 ROI
void mouse_callback(int event, int x, int y, int, void*)
{
    switch (event)
    {
        case cv::EVENT_MOUSEMOVE:
            if (drawing_box)
            {
                box.width = x - box.x;
                box.height = y - box.y;
            }
            break;
        case cv::EVENT_LBUTTONDOWN:
            drawing_box = true;
            box = cv::Rect(x, y, 0, 0);
            break;
        case cv::EVENT_LBUTTONUP:
            drawing_box = false;
			change_template = true;
            if (box.width < 0)
            {
                box.x += box.width;
                box.width *= -1;
            }
            if (box.height < 0)
            {
                box.y += box.height;
                box.height *= -1;
            }
            break;
    }
}

int main(int argc, char** argv)
{
	// 匹配器参数
	template_matching::MatcherParam param;
	param.angle = 0;
	param.iouThreshold = 0;
	param.matcherType = template_matching::MatcherType::PATTERN;
	param.maxCount = 1;
	param.minArea = 256;
	param.scoreThreshold = 0.5;
	
	// 匹配结果
	std::vector<template_matching::MatchResult> matchResults;

#ifdef _WIN32
	// windows 动态加载dll
	HINSTANCE handle = nullptr;
	handle = LoadLibrary("templatematching.dll");
	if (handle == nullptr)
	{
		std::cerr << "Error : failed to load templatematching.dll!" << std::endl;
		return -2;
	}

	// 获取动态库内的函数
	InitMD myGetMatcher;
	myGetMatcher = (InitMD)GetProcAddress(handle, "GetMatcher");
#else
	// linux 动态加载dll
	void* handle = nullptr;
	handle = dlopen("libtemplatematching.so", RTLD_LAZY); 
	if (handle == nullptr)
	{
		std::cerr << "Error : failed to load libtemplatematching.so!" << std::endl;
		return -2;
	}

	// 获取动态库内的函数
	InitMD myGetMatcher;
	myGetMatcher = (InitMD)dlsym(handle, "GetMatcher");
#endif // _WIN32
	// std::cout << "Load getDetector." << std::endl;

	// 初始化匹配器
	std::cout << "initiating..." << std::endl;
	template_matching::Matcher* matcher = myGetMatcher(param);
	std::cout << "initialized." << std::endl;

	if (matcher)
	{
		matcher->setMetricsTime(true);

		std::chrono::steady_clock::time_point startTime, endTime;
		std::chrono::duration<double> timeUse;

		// 打开0号摄像头
		cv::VideoCapture cap(0);
		if (!cap.isOpened())
		{
			std::cerr << "Error: failed to open camera." << std::endl;
			return -1;
		}
		// 设置摄像头分辨率
		//cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
		//cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
		// 设置摄像头编码
		//cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

		cv::namedWindow("frame");
		cv::setMouseCallback("frame", mouse_callback);

		// 读取摄像头图像
		cv::Mat frame, drawFrame, roi;
		while (true)
		{
			cap >> frame;
			if (frame.empty())
			{
				std::cerr << "Error: failed to read frame." << std::endl;
				break;
			}

			cv::Mat drawFrame = frame.clone();
			if (drawFrame.channels() == 1)
			{
				cv::cvtColor(drawFrame, drawFrame, cv::COLOR_GRAY2BGR);
			}

			if (drawing_box)
				cv::rectangle(drawFrame, box, cv::Scalar(0, 0, 255), 2);
			else if (box.area() > 0)
			{
				if (change_template)
				{
					//cv::rectangle(drawFrame, box, cv::Scalar(0, 255, 0), 2);
					roi = frame(box);
					//cv::imshow("ROI", roi);
					cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);
					matcher->setTemplate(roi);
					change_template = false;
				}
				
			}

			startTime = std::chrono::steady_clock::now();

			// 执行匹配，结果保存在 matchResults
			cv::Mat frame_gray;
			cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
			matcher->match(frame_gray, matchResults);

			endTime = std::chrono::steady_clock::now();
			timeUse = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
			std::cout << "match time: " << timeUse.count() << "s." << std::endl;

			// 可视化查看
			
			for (int i = 0; i < matchResults.size(); i++)
			{
				cv::Point2i temp;
				std::vector<cv::Point2i> pts;
				temp.x = std::round(matchResults[i].LeftTop.x);
				temp.y = std::round(matchResults[i].LeftTop.y);
				pts.push_back(temp);
				temp.x = std::round(matchResults[i].RightTop.x);
				temp.y = std::round(matchResults[i].RightTop.y);
				pts.push_back(temp);
				temp.x = std::round(matchResults[i].RightBottom.x);
				temp.y = std::round(matchResults[i].RightBottom.y);
				pts.push_back(temp);
				temp.x = std::round(matchResults[i].LeftBottom.x);
				temp.y = std::round(matchResults[i].LeftBottom.y);
				pts.push_back(temp);

				cv::polylines(drawFrame, pts, true, cv::Scalar(0, 255, 0), 1, cv::LINE_8);
			}
				

			// 显示图像，按ESC或q退出
			cv::imshow("frame", drawFrame);
			char key = cv::waitKey(1);
			if (key == 27 || key == 'q' || key == 'Q')
			{
				break;
			}
		}

		cap.release();

		delete matcher;
		
	}
	else
	{
		std::cerr << "Error: failed to get matcher." << std::endl;
	}

	// 释放 windows 动态库
	if (handle != nullptr)
	{
#ifdef _WIN32
		FreeLibrary(handle);
#else
		dlclose(handle);
#endif // _WIN32
	}

	return 0;
}
