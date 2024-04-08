#include "base_matcher.h"

namespace template_matching
{

	BaseMatcher::BaseMatcher()
	{
		logger_ = spdlog::get("template_matching");
		logger_->set_level(spdlog::level::info);
		
	}

	BaseMatcher::~BaseMatcher()
	{
		// std::cout << "~BaseMatcher()" << std::endl;
	}

	bool BaseMatcher::isInited()
	{
		return initFinishedFlag_;
	}

	bool BaseMatcher::initMatcher(const template_matching::MatcherParam& param)
	{
		matchParam_ = param;

		return true;
	}

	void BaseMatcher::setMetricsTime(const bool& enabled)
	{
		metricsTime_ = enabled;
	}

	bool BaseMatcher::getMetricsTime()
	{
		return metricsTime_;
	}

	void BaseMatcher::drawResult(const cv::Mat& frame, std::vector<template_matching::MatchResult> matchResults)
	{
		if (frame.empty())
		{
			logger_->warn("image empty.");
			return;
		}

		cv::Mat drawFrame = frame.clone();
		if (drawFrame.channels() == 1)
		{
			cv::cvtColor(drawFrame, drawFrame, cv::COLOR_GRAY2BGR);
		}
		
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

		cv::imwrite("demo.png", drawFrame);
		cv::imshow("demo", drawFrame);
		cv::waitKey();
	}
}
