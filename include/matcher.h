#ifndef _TEMPLATE_MATCHER_H
#define _TEMPLATE_MATCHER_H  

#include <vector>
#include "template_matching.h"

namespace template_matching
{

	class LIB_API Matcher
	{
	public:
		~Matcher() {  }

		/** 匹配
		@param frame 输入图像.
		@param matchResults 匹配结果.
		*/
		virtual int match(const cv::Mat &frame, std::vector<MatchResult> & matchResults) = 0;

		/** 设置模板
		@param templateImage 模板图像.
		*/
		virtual int setTemplate(const cv::Mat& templateImage) = 0;

		/** 绘制匹配结果
		@param frame 输入图像.
		@param matchResults 匹配结果.
		*/
		virtual void drawResult(const cv::Mat& frame, std::vector<MatchResult> matchResults) = 0;

		/** 设置是否打印中间运行时间
		@param enabled 启用标志位.
		*/
		virtual void setMetricsTime(const bool& enabled) = 0;

		/** 获取是否打印中间运行时间标志位
		*/
		virtual bool getMetricsTime() = 0;

	protected:
	
	};

	/** 获取匹配器
	@param param 配置参数.
	*/
	extern "C" LIB_API Matcher * GetMatcher(const template_matching::MatcherParam& param);
	
}

#endif