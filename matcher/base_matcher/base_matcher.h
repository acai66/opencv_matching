#ifndef _BASEMATCHER_H 
#define _BASEMATCHER_H 

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <random>
#include <chrono>
#include "matcher.h"

#include "omp.h"


namespace template_matching
{
	class BaseMatcher : public Matcher
	{
	public:
		BaseMatcher();

		~BaseMatcher();

		// 判断是否初始化成功
		bool isInited();

		virtual void drawResult(const cv::Mat& frame, std::vector<template_matching::MatchResult> matchResults);

		void setMetricsTime(const bool& enabled);

		bool getMetricsTime();

	protected:

		// 初始化匹配器
		bool initMatcher(const MatcherParam& param);

		// 初始化完成标志位
		bool initFinishedFlag_ = false;

		// 匹配参数
		MatcherParam matchParam_;

		// template
		cv::Mat templateImage_;

		// 计时
		std::chrono::steady_clock::time_point startTime_, endTime_;
		std::chrono::duration<double> timeUse_;

		// 评估时间
		bool metricsTime_ = false;

		std::shared_ptr<spdlog::logger> logger_;


	private:


	};
}

#endif
