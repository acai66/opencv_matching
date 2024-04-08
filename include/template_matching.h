#ifndef _TEMPLATE_MATCHING_DEFINE_H 
#define _TEMPLATE_MATCHING_DEFINE_H 

#pragma once

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
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>

namespace template_matching
{
	// 匹配器类型
	enum LIB_API MatcherType { 
		PATTERN = 0,

	};

	// 匹配器参数
	struct LIB_API MatcherParam
	{
		// 匹配器类型
		MatcherType matcherType = MatcherType::PATTERN;

		// 最大匹配目标数量
		int maxCount = 200;

		// 匹配得分阈值
		double scoreThreshold = 0.5;

		// 重叠框去重iou阈值
		double iouThreshold = 0.0;

		// 匹配角度范围
		double angle = 0;

		// 顶层金字塔最小面积
		double minArea = 256;

	};
	
	// 匹配结果
	struct LIB_API MatchResult
	{
		cv::Point2d LeftTop; // 左上角点
		cv::Point2d LeftBottom; // 左下角点
		cv::Point2d RightTop; // 右上角点
		cv::Point2d RightBottom; // 右下角点
		cv::Point2d Center; // 中心点

		double Angle; // 角度
		double Score; // 匹配得分
	};

}

#endif
