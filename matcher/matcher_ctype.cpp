#ifdef _WIN32
#pragma warning(disable: 4251)
#endif  // _WIN32

#ifdef LIB_SHARED_BUILD
#ifdef _WIN32
#ifdef LIB_EXPORTS
#define LIB_API __declspec(dllexport)
#else
#define LIB_API __declspec(dllimport)
#endif  // MY_LIB_EXPORTS
#else
#define LIB_API
#endif  // _WIN32
#else
#define LIB_API
#endif  // MY_LIB_SHARED_BUILD

#include <opencv2/opencv.hpp>

#include "../include/matcher.h"
#include "../include/template_matching.h"

#include "matcher.h"
#ifdef _WIN32
#include "windows.h"
#else
#include <dlfcn.h>
#endif // _WIN32


// 动态库函数指针
typedef template_matching::Matcher* (*InitMD)(const template_matching::MatcherParam&);


// 匹配结果
struct LIB_API MatchResult
{
    double leftTopX;
    double leftTopY;
    double leftBottomX;
    double leftBottomY;
    double rightTopX;
    double rightTopY;
    double rightBottomX;
    double rightBottomY;
    double centerX;
    double centerY;
    double angle;
    double score;
};


class Matcher
{
public:
	Matcher(const template_matching::MatcherParam& param);
	~Matcher();
	virtual int match(const cv::Mat& frame, MatchResult* Results, int maxCount);
	virtual int setTemplate(const cv::Mat& frame);

protected:


private:

#ifdef WIN32
	// windows 动态加载dll
	HINSTANCE _handle = nullptr;
#else
	void* _handle = nullptr;
#endif

	InitMD _myGetMatcher = nullptr;
	template_matching::Matcher* _matcher = nullptr;
};

Matcher::Matcher(const template_matching::MatcherParam& param)
{

#ifdef WIN32
	// windows 动态加载dll
	_handle = LoadLibrary("templatematching.dll");
	if (_handle == nullptr)
	{
		std::cerr << "Error : failed to load templatematching.dll!" << std::endl;
		return;
	}

	// 获取动态库内的函数
	_myGetMatcher = (InitMD)GetProcAddress(_handle, "GetMatcher");
#else
	// linux 动态加载dll
	_handle = dlopen("libtemplatematching.so", RTLD_LAZY);
	if (_handle == nullptr)
	{
		std::cerr << "Error : failed to load libtemplatematching.so!" << std::endl;
		return;
	}

	// 获取动态库内的函数
	_myGetMatcher = (InitMD)dlsym(_handle, "GetMatcher");
#endif

	if (_myGetMatcher == nullptr)
	{
		std::cerr << "Error : failed to load getDetector!" << std::endl;
		return;
	}

	_matcher = _myGetMatcher(param);
	_matcher->setMetricsTime(false);

}

Matcher::~Matcher()
{
	if (_matcher != nullptr)
	{
		delete _matcher;
	}

	if (_handle != nullptr)
	{
#ifdef _WIN32
		FreeLibrary(_handle);
#else
		dlclose(_handle);
#endif // _WIN32
	}
}

int Matcher::setTemplate(const cv::Mat& frame)
{
	int returnSize = 0;
	if (_matcher)
	{
		returnSize = _matcher->setTemplate(frame);
	}
	else
	{
		returnSize = -4;
	}

	return returnSize;
}


int Matcher::match(const cv::Mat& frame, MatchResult* Results, int maxCount)
{
	// 匹配结果
	std::vector<template_matching::MatchResult> matchResults;

	if (_matcher)
	{
		int returnSize = _matcher->match(frame, matchResults);
		//std::cout << "Lib: returnSize: " << returnSize << std::endl;
		if (returnSize < 0)
		{
			return returnSize;
		}

		returnSize = matchResults.size() < maxCount ? matchResults.size() : maxCount;



		for (int i = 0; i < returnSize; i++)
		{
			(Results + i)->leftTopX = matchResults[i].LeftTop.x;
			(Results + i)->leftTopY = matchResults[i].LeftTop.y;
			(Results + i)->leftBottomX = matchResults[i].LeftBottom.x;
			(Results + i)->leftBottomY = matchResults[i].LeftBottom.y;
			(Results + i)->rightTopX = matchResults[i].RightTop.x;
			(Results + i)->rightTopY = matchResults[i].RightTop.y;
			(Results + i)->rightBottomX = matchResults[i].RightBottom.x;
			(Results + i)->rightBottomY = matchResults[i].RightBottom.y;
			(Results + i)->centerX = matchResults[i].Center.x;
			(Results + i)->centerY = matchResults[i].Center.y;
			(Results + i)->angle = matchResults[i].Angle;
			(Results + i)->score = matchResults[i].Score;
		}

		return returnSize;
	}
	else
	{
		return -3;
	}
}

// 对外提供dll功能接口：初始化匹配器
extern"C" LIB_API Matcher * matcher(int maxCount, float scoreThreshold, float iouThreshold, float angle, float minArea)
{
	template_matching::MatcherParam param;
	param.matcherType = template_matching::MatcherType::PATTERN;
	param.maxCount = maxCount;
	param.scoreThreshold = scoreThreshold;
	param.iouThreshold = iouThreshold;
	param.angle = angle;
	param.minArea = minArea;

	Matcher* obj1 = new Matcher(param);

	return obj1;
}

// 对外提供dll功能接口：设置模板
extern"C" LIB_API int setTemplate(Matcher * obj1, uchar * data, int width, int height, int channels)
{
	cv::Mat img;

	if (channels == 1)
	{
		img = cv::Mat(cv::Size(width, height), CV_8UC1, data);
	}
	else
	{
		std::cerr << "Error : not support channels: " << channels << std::endl;
		return -1;
	}

	int returnSize = obj1->setTemplate(img);

	return returnSize;

}

// 对外提供dll功能接口：执行匹配
extern"C" LIB_API int match(Matcher * obj1, uchar * data, int width, int height, int channels, MatchResult * Results, int maxCount)
{
	cv::Mat img;

	if (channels == 1)
	{
		img = cv::Mat(cv::Size(width, height), CV_8UC1, data);
	}
	else
	{
		std::cerr << "Error : not support channels: " << channels << std::endl;
		return -1;
	}

	int returnSize = obj1->match(img, Results, maxCount);

	return returnSize;

}
