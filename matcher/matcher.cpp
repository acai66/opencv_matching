#include "base_matcher/base_matcher.h"
#include "Pattern_Matching/PatternMatching.h"

#include <fstream>

namespace template_matching
{
	auto logger_ = spdlog::stdout_color_mt("template_matching");  //spdlog::rotating_logger_mt<spdlog::async_factory>("file_logger", "run.log", 1024 * 1024 * 5, 1);

	Matcher* GetMatcher(const MatcherParam& param)
	{
		MatcherParam paramCopy = param;
		BaseMatcher* matcher = nullptr;
		
		switch (paramCopy.matcherType)
		{
		case MatcherType::PATTERN:
			logger_->info("Initializing matcher for type: PATTERN");
			matcher = new PatternMatcher(paramCopy);
			break;
		default:
			break;
		}

		if (matcher->isInited() == false)
		{
			delete matcher;
			matcher = nullptr;
		}

		return matcher;
	}
	
}