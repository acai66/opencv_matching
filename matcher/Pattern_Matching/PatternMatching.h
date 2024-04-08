#ifndef _PATTERNMATCHING_H 
#define _PATTERNMATCHING_H 
#pragma once

#include "../base_matcher/base_matcher.h"
#include "template_matching.h"
#include <ctime>
#include <assert.h>
#include <numeric>
#include <vector>

#define VISION_TOLERANCE 0.0000001
#define D2R (CV_PI / 180.0)
#define R2D (180.0 / CV_PI)
#define MATCH_CANDIDATE_NUM 5


namespace template_matching
{
	using namespace cv;
	using namespace std;

	struct s_TemplData
	{
		vector<Mat> vecPyramid;
		vector<Scalar> vecTemplMean;
		vector<double> vecTemplNorm;
		vector<double> vecInvArea;
		vector<bool> vecResultEqual1;
		bool bIsPatternLearned;
		int iBorderColor;
		void clear()
		{
			vector<Mat>().swap(vecPyramid);
			vector<double>().swap(vecTemplNorm);
			vector<double>().swap(vecInvArea);
			vector<Scalar>().swap(vecTemplMean);
			vector<bool>().swap(vecResultEqual1);
		}
		void resize(int iSize)
		{
			vecTemplMean.resize(iSize);
			vecTemplNorm.resize(iSize, 0);
			vecInvArea.resize(iSize, 1);
			vecResultEqual1.resize(iSize, false);
		}
		s_TemplData()
		{
			bIsPatternLearned = false;
		}
	};
	struct s_MatchParameter
	{
		Point2d pt;
		double dMatchScore;
		double dMatchAngle;
		//Mat matRotatedSrc;
		Rect rectRoi;
		double dAngleStart;
		double dAngleEnd;
		RotatedRect rectR;
		Rect rectBounding;
		bool bDelete;

		double vecResult[3][3];//for subpixel
		int iMaxScoreIndex;//for subpixel
		bool bPosOnBorder;
		Point2d ptSubPixel;
		double dNewAngle;

		s_MatchParameter(Point2f ptMinMax, double dScore, double dAngle)//, Mat matRotatedSrc = Mat ())
		{
			pt = ptMinMax;
			dMatchScore = dScore;
			dMatchAngle = dAngle;

			bDelete = false;
			dNewAngle = 0.0;

			bPosOnBorder = false;
		}
		s_MatchParameter()
		{
			double dMatchScore = 0;
			double dMatchAngle = 0;
		}
		~s_MatchParameter()
		{

		}
	};
	struct s_BlockMax
	{
		struct Block
		{
			Rect rect;
			double dMax;
			Point ptMaxLoc;
			Block()
			{}
			Block(Rect rect_, double dMax_, Point ptMaxLoc_)
			{
				rect = rect_;
				dMax = dMax_;
				ptMaxLoc = ptMaxLoc_;
			}
		};
		s_BlockMax()
		{}
		vector<Block> vecBlock;
		Mat matSrc;
		s_BlockMax(Mat matSrc_, Size sizeTemplate)
		{
			matSrc = matSrc_;
			//將matSrc 拆成數個block，分別計算最大值
			int iBlockW = sizeTemplate.width * 2;
			int iBlockH = sizeTemplate.height * 2;

			int iCol = matSrc.cols / iBlockW;
			bool bHResidue = matSrc.cols % iBlockW != 0;

			int iRow = matSrc.rows / iBlockH;
			bool bVResidue = matSrc.rows % iBlockH != 0;

			if (iCol == 0 || iRow == 0)
			{
				vecBlock.clear();
				return;
			}

			vecBlock.resize(iCol * iRow);
			int iCount = 0;
			for (int y = 0; y < iRow; y++)
			{
				for (int x = 0; x < iCol; x++)
				{
					Rect rectBlock(x * iBlockW, y * iBlockH, iBlockW, iBlockH);
					vecBlock[iCount].rect = rectBlock;
					minMaxLoc(matSrc(rectBlock), 0, &vecBlock[iCount].dMax, 0, &vecBlock[iCount].ptMaxLoc);
					vecBlock[iCount].ptMaxLoc += rectBlock.tl();
					iCount++;
				}
			}
			if (bHResidue && bVResidue)
			{
				Rect rectRight(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
				Block blockRight;
				blockRight.rect = rectRight;
				minMaxLoc(matSrc(rectRight), 0, &blockRight.dMax, 0, &blockRight.ptMaxLoc);
				blockRight.ptMaxLoc += rectRight.tl();
				vecBlock.push_back(blockRight);

				Rect rectBottom(0, iRow * iBlockH, iCol * iBlockW, matSrc.rows - iRow * iBlockH);
				Block blockBottom;
				blockBottom.rect = rectBottom;
				minMaxLoc(matSrc(rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
				blockBottom.ptMaxLoc += rectBottom.tl();
				vecBlock.push_back(blockBottom);
			}
			else if (bHResidue)
			{
				Rect rectRight(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
				Block blockRight;
				blockRight.rect = rectRight;
				minMaxLoc(matSrc(rectRight), 0, &blockRight.dMax, 0, &blockRight.ptMaxLoc);
				blockRight.ptMaxLoc += rectRight.tl();
				vecBlock.push_back(blockRight);
			}
			else
			{
				Rect rectBottom(0, iRow * iBlockH, matSrc.cols, matSrc.rows - iRow * iBlockH);
				Block blockBottom;
				blockBottom.rect = rectBottom;
				minMaxLoc(matSrc(rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
				blockBottom.ptMaxLoc += rectBottom.tl();
				vecBlock.push_back(blockBottom);
			}
		}
		void UpdateMax(Rect rectIgnore)
		{
			if (vecBlock.size() == 0)
				return;
			//找出所有跟rectIgnore交集的block
			int iSize = vecBlock.size();
			for (int i = 0; i < iSize; i++)
			{
				Rect rectIntersec = rectIgnore & vecBlock[i].rect;
				//無交集
				if (rectIntersec.width == 0 && rectIntersec.height == 0)
					continue;
				//有交集，更新極值和極值位置
				minMaxLoc(matSrc(vecBlock[i].rect), 0, &vecBlock[i].dMax, 0, &vecBlock[i].ptMaxLoc);
				vecBlock[i].ptMaxLoc += vecBlock[i].rect.tl();
			}
		}
		void GetMaxValueLoc(double& dMax, Point& ptMaxLoc)
		{
			int iSize = vecBlock.size();
			if (iSize == 0)
			{
				minMaxLoc(matSrc, 0, &dMax, 0, &ptMaxLoc);
				return;
			}
			//從block中找最大值
			int iIndex = 0;
			dMax = vecBlock[0].dMax;
			for (int i = 1; i < iSize; i++)
			{
				if (vecBlock[i].dMax >= dMax)
				{
					iIndex = i;
					dMax = vecBlock[i].dMax;
				}
			}
			ptMaxLoc = vecBlock[iIndex].ptMaxLoc;
		}
	};

	class PatternMatcher : public BaseMatcher
	{
	public:
		PatternMatcher(const template_matching::MatcherParam& param);
		~PatternMatcher();
		virtual int match(const cv::Mat & frame, std::vector<template_matching::MatchResult> &matchResults) override;

		virtual int setTemplate(const cv::Mat& templateImage) override;
		
	
	protected:
		

	private:
		s_TemplData m_TemplData;
		bool m_bDebugMode = false;
		bool m_bSubPixel = true;
		bool m_bStopLayer1 = false;
	
	};

	
}

#endif
