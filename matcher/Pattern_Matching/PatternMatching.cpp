#include "PatternMatching.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif


namespace template_matching {

	int GetTopLayer(Mat* matTempl, int iMinDstLength)
	{
		int iTopLayer = 0;
		int iMinReduceArea = iMinDstLength * iMinDstLength;
		int iArea = matTempl->cols * matTempl->rows;
		while (iArea > iMinReduceArea)
		{
			iArea /= 4;
			iTopLayer++;
		}
		return iTopLayer;
	}

	void LearnPattern(s_TemplData& m_TemplData, Mat& m_matDst, double& m_iMinReduceArea)
	{
		m_TemplData.clear();

		int iTopLayer = GetTopLayer(&m_matDst, (int)sqrt((double)m_iMinReduceArea));
		buildPyramid(m_matDst, m_TemplData.vecPyramid, iTopLayer);
		s_TemplData* templData = &m_TemplData;
		templData->iBorderColor = mean(m_matDst).val[0] < 128 ? 255 : 0;
		int iSize = templData->vecPyramid.size();
		templData->resize(iSize);

		for (int i = 0; i < iSize; i++)
		{
			double invArea = 1. / ((double)templData->vecPyramid[i].rows * templData->vecPyramid[i].cols);
			Scalar templMean, templSdv;
			double templNorm = 0, templSum2 = 0;

			meanStdDev(templData->vecPyramid[i], templMean, templSdv);
			templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

			if (templNorm < DBL_EPSILON)
			{
				templData->vecResultEqual1[i] = true;
			}
			templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] + templMean[2] * templMean[2] + templMean[3] * templMean[3];


			templSum2 /= invArea;
			templNorm = std::sqrt(templNorm);
			templNorm /= std::sqrt(invArea); // care of accuracy here


			templData->vecInvArea[i] = invArea;
			templData->vecTemplMean[i] = templMean;
			templData->vecTemplNorm[i] = templNorm;
		}
		templData->bIsPatternLearned = true;
	}

	Point2f ptRotatePt2f(Point2f ptInput, Point2f ptOrg, double dAngle)
	{
		double dWidth = ptOrg.x * 2;
		double dHeight = ptOrg.y * 2;
		double dY1 = dHeight - ptInput.y, dY2 = dHeight - ptOrg.y;

		double dX = (ptInput.x - ptOrg.x) * cos(dAngle) - (dY1 - ptOrg.y) * sin(dAngle) + ptOrg.x;
		double dY = (ptInput.x - ptOrg.x) * sin(dAngle) + (dY1 - ptOrg.y) * cos(dAngle) + dY2;

		dY = -dY + dHeight;
		return Point2f((float)dX, (float)dY);
	}

	Size GetBestRotationSize(Size sizeSrc, Size sizeDst, double dRAngle)
	{
		double dRAngle_radian = dRAngle * D2R;
		Point ptLT(0, 0), ptLB(0, sizeSrc.height - 1), ptRB(sizeSrc.width - 1, sizeSrc.height - 1), ptRT(sizeSrc.width - 1, 0);
		Point2f ptCenter((sizeSrc.width - 1) / 2.0f, (sizeSrc.height - 1) / 2.0f);
		Point2f ptLT_R = ptRotatePt2f(Point2f(ptLT), ptCenter, dRAngle_radian);
		Point2f ptLB_R = ptRotatePt2f(Point2f(ptLB), ptCenter, dRAngle_radian);
		Point2f ptRB_R = ptRotatePt2f(Point2f(ptRB), ptCenter, dRAngle_radian);
		Point2f ptRT_R = ptRotatePt2f(Point2f(ptRT), ptCenter, dRAngle_radian);

		float fTopY = max(max(ptLT_R.y, ptLB_R.y), max(ptRB_R.y, ptRT_R.y));
		float fBottomY = min(min(ptLT_R.y, ptLB_R.y), min(ptRB_R.y, ptRT_R.y));
		float fRightX = max(max(ptLT_R.x, ptLB_R.x), max(ptRB_R.x, ptRT_R.x));
		float fLeftX = min(min(ptLT_R.x, ptLB_R.x), min(ptRB_R.x, ptRT_R.x));

		if (dRAngle > 360)
			dRAngle -= 360;
		else if (dRAngle < 0)
			dRAngle += 360;

		if (fabs(fabs(dRAngle) - 90) < VISION_TOLERANCE || fabs(fabs(dRAngle) - 270) < VISION_TOLERANCE)
		{
			return Size(sizeSrc.height, sizeSrc.width);
		}
		else if (fabs(dRAngle) < VISION_TOLERANCE || fabs(fabs(dRAngle) - 180) < VISION_TOLERANCE)
		{
			return sizeSrc;
		}

		double dAngle = dRAngle;

		if (dAngle > 0 && dAngle < 90)
		{
			;
		}
		else if (dAngle > 90 && dAngle < 180)
		{
			dAngle -= 90;
		}
		else if (dAngle > 180 && dAngle < 270)
		{
			dAngle -= 180;
		}
		else if (dAngle > 270 && dAngle < 360)
		{
			dAngle -= 270;
		}
		else//Debug
		{
			cout << "Unkown" << endl;;
		}

		float fH1 = sizeDst.width * sin(dAngle * D2R) * cos(dAngle * D2R);
		float fH2 = sizeDst.height * sin(dAngle * D2R) * cos(dAngle * D2R);

		int iHalfHeight = (int)ceil(fTopY - ptCenter.y - fH1);
		int iHalfWidth = (int)ceil(fRightX - ptCenter.x - fH2);

		Size sizeRet(iHalfWidth * 2, iHalfHeight * 2);

		bool bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height)
			|| (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height
				|| sizeDst.area() > sizeRet.area());
		if (bWrongSize)
			sizeRet = Size(int(fRightX - fLeftX + 0.5), int(fTopY - fBottomY + 0.5));

		return sizeRet;
	}

#ifdef __aarch64__
/*
	inline int32_t vaddvq_s32(int32x4_t v) {
		int32x2_t tmp = vpadd_s32(vget_low_s32(v), vget_high_s32(v));
		return vget_lane_s32(vpadd_s32(tmp, tmp), 0);
	}

	inline int32_t IM_Conv_SIMD(uint8_t* pCharKernel, uint8_t* pCharConv, int iLength) {
		const int iBlockSize = 16, Block = iLength / iBlockSize;
		int32x4_t SumV = vdupq_n_s32(0);
		uint8x16_t Zero = vdupq_n_u8(0);

		for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize) {
			uint8x16_t SrcK = vld1q_u8(pCharKernel + Y);
			uint8x16_t SrcC = vld1q_u8(pCharConv + Y);
			int16x8_t SrcK_L = vmovl_u8(vget_low_u8(SrcK));
			int16x8_t SrcK_H = vmovl_u8(vget_high_u8(SrcK));
			int16x8_t SrcC_L = vmovl_u8(vget_low_u8(SrcC));
			int16x8_t SrcC_H = vmovl_u8(vget_high_u8(SrcC));
			int32x4_t SumT = vaddq_s32(vmull_s16(vget_low_s16(SrcK_L), vget_low_s16(SrcC_L)), vmull_s16(vget_low_s16(SrcK_H), vget_low_s16(SrcC_H)));
			SumV = vaddq_s32(SumV, SumT);
		}

		int32_t Sum = vaddvq_s32(SumV);

		for (int Y = Block * iBlockSize; Y < iLength; Y++) {
			Sum += pCharKernel[Y] * pCharConv[Y];
		}

		return Sum;
	}
*/
	inline uint32_t IM_Conv_SIMD(uint8_t* pCharKernel, uint8_t* pCharConv, int iLength) {
		uint32_t Sum = 0;

		for (int Y = 0; Y < iLength; Y++) {
			Sum += pCharKernel[Y] * pCharConv[Y];
		}

		return Sum;
	}
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_X64)
	//From ImageShop
	// 4個有符號的32位的數據相加的和。
	inline int _mm_hsum_epi32(__m128i V)      // V3 V2 V1 V0
	{
		// 實測這個速度要快些，_mm_extract_epi32最慢。
		__m128i T = _mm_add_epi32(V, _mm_srli_si128(V, 8));  // V3+V1   V2+V0  V1  V0  
		T = _mm_add_epi32(T, _mm_srli_si128(T, 4));    // V3+V1+V2+V0  V2+V0+V1 V1+V0 V0 
		return _mm_cvtsi128_si32(T);       // 提取低位 
	}
	// 基於SSE的字節數據的乘法。
	// <param name="Kernel">需要卷積的核矩陣。 </param>
	// <param name="Conv">卷積矩陣。 </param>
	// <param name="Length">矩陣所有元素的長度。 </param>
	inline int IM_Conv_SIMD(unsigned char* pCharKernel, unsigned char* pCharConv, int iLength)
	{
		const int iBlockSize = 16, Block = iLength / iBlockSize;
		__m128i SumV = _mm_setzero_si128();
		__m128i Zero = _mm_setzero_si128();
		for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize)
		{
			__m128i SrcK = _mm_loadu_si128((__m128i*)(pCharKernel + Y));
			__m128i SrcC = _mm_loadu_si128((__m128i*)(pCharConv + Y));
			__m128i SrcK_L = _mm_unpacklo_epi8(SrcK, Zero);
			__m128i SrcK_H = _mm_unpackhi_epi8(SrcK, Zero);
			__m128i SrcC_L = _mm_unpacklo_epi8(SrcC, Zero);
			__m128i SrcC_H = _mm_unpackhi_epi8(SrcC, Zero);
			__m128i SumT = _mm_add_epi32(_mm_madd_epi16(SrcK_L, SrcC_L), _mm_madd_epi16(SrcK_H, SrcC_H));
			SumV = _mm_add_epi32(SumV, SumT);
		}
		int Sum = _mm_hsum_epi32(SumV);
		for (int Y = Block * iBlockSize; Y < iLength; Y++)
		{
			Sum += pCharKernel[Y] * pCharConv[Y];
		}
		return Sum;
	}
#else
	// Other architecture
#endif


	void CCOEFF_Denominator(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer)
	{
		if (pTemplData->vecResultEqual1[iLayer])
		{
			matResult = Scalar::all(1);
			return;
		}
		double* q0 = 0, * q1 = 0, * q2 = 0, * q3 = 0;

		Mat sum, sqsum;
		integral(matSrc, sum, sqsum, CV_64F);

		q0 = (double*)sqsum.data;
		q1 = q0 + pTemplData->vecPyramid[iLayer].cols;
		q2 = (double*)(sqsum.data + pTemplData->vecPyramid[iLayer].rows * sqsum.step);
		q3 = q2 + pTemplData->vecPyramid[iLayer].cols;

		double* p0 = (double*)sum.data;
		double* p1 = p0 + pTemplData->vecPyramid[iLayer].cols;
		double* p2 = (double*)(sum.data + pTemplData->vecPyramid[iLayer].rows * sum.step);
		double* p3 = p2 + pTemplData->vecPyramid[iLayer].cols;

		int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
		int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;

		//
		double dTemplMean0 = pTemplData->vecTemplMean[iLayer][0];
		double dTemplNorm = pTemplData->vecTemplNorm[iLayer];
		double dInvArea = pTemplData->vecInvArea[iLayer];
		//

		int i, j;
		for (i = 0; i < matResult.rows; i++)
		{
			float* rrow = matResult.ptr<float>(i);
			int idx = i * sumstep;
			int idx2 = i * sqstep;

			for (j = 0; j < matResult.cols; j += 1, idx += 1, idx2 += 1)
			{
				double num = rrow[j], t;
				double wndMean2 = 0, wndSum2 = 0;

				t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
				wndMean2 += t * t;
				num -= t * dTemplMean0;
				wndMean2 *= dInvArea;


				t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
				wndSum2 += t;


				//t = std::sqrt (MAX (wndSum2 - wndMean2, 0)) * dTemplNorm;

				double diff2 = MAX(wndSum2 - wndMean2, 0);
				if (diff2 <= std::min(0.5, 10 * FLT_EPSILON * wndSum2))
					t = 0; // avoid rounding errors
				else
					t = std::sqrt(diff2) * dTemplNorm;

				if (fabs(num) < t)
					num /= t;
				else if (fabs(num) < t * 1.125)
					num = num > 0 ? 1 : -1;
				else
					num = 0;

				rrow[j] = (float)num;
			}
		}
	}

	void MatchTemplate(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, bool bUseSIMD)
	{
		if (bUseSIMD)
		{
			//From ImageShop
			matResult.create(matSrc.rows - pTemplData->vecPyramid[iLayer].rows + 1,
				matSrc.cols - pTemplData->vecPyramid[iLayer].cols + 1, CV_32FC1);
			matResult.setTo(0);
			cv::Mat& matTemplate = pTemplData->vecPyramid[iLayer];

			int  t_r_end = matTemplate.rows, t_r = 0;
			for (int r = 0; r < matResult.rows; r++)
			{
				float* r_matResult = matResult.ptr<float>(r);
				uchar* r_source = matSrc.ptr<uchar>(r);
				uchar* r_template, * r_sub_source;
				for (int c = 0; c < matResult.cols; ++c, ++r_matResult, ++r_source)
				{
					r_template = matTemplate.ptr<uchar>();
					r_sub_source = r_source;
					for (t_r = 0; t_r < t_r_end; ++t_r, r_sub_source += matSrc.cols, r_template += matTemplate.cols)
					{
						*r_matResult = *r_matResult + IM_Conv_SIMD(r_template, r_sub_source, matTemplate.cols);
					}
				}
			}
			//From ImageShop
		}
		else
			matchTemplate(matSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCORR);

		/*Mat diff;
		absdiff(matResult, matResult, diff);
		double dMaxValue;
		minMaxLoc(diff, 0, &dMaxValue, 0,0);*/
		CCOEFF_Denominator(matSrc, pTemplData, matResult, iLayer);
	}

	Point GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap)
	{
		//比對到的區域完全不重疊 : +-一個樣板寬高
		//int iStartX = ptMaxLoc.x - iTemplateW;
		//int iStartY = ptMaxLoc.y - iTemplateH;
		//int iEndX = ptMaxLoc.x + iTemplateW;

		//int iEndY = ptMaxLoc.y + iTemplateH;
		////塗黑
		//rectangle (matResult, Rect (iStartX, iStartY, 2 * iTemplateW * (1-dMaxOverlap * 2), 2 * iTemplateH * (1-dMaxOverlap * 2)), Scalar (dMinValue), CV_FILLED);
		////得到下一個最大值
		//Point ptNewMaxLoc;
		//minMaxLoc (matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
		//return ptNewMaxLoc;

		//比對到的區域需考慮重疊比例
		int iStartX = ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap);
		int iStartY = ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap);
		//塗黑
		rectangle(matResult, Rect(iStartX, iStartY, 2 * sizeTemplate.width * (1 - dMaxOverlap), 2 * sizeTemplate.height * (1 - dMaxOverlap)), Scalar(-1), CV_FILLED);
		//得到下一個最大值
		Point ptNewMaxLoc;
		minMaxLoc(matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
		return ptNewMaxLoc;
	}

	Point GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap, s_BlockMax& blockMax)
	{
		//比對到的區域需考慮重疊比例
		int iStartX = int(ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap));
		int iStartY = int(ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap));
		Rect rectIgnore(iStartX, iStartY, int(2 * sizeTemplate.width * (1 - dMaxOverlap))
			, int(2 * sizeTemplate.height * (1 - dMaxOverlap)));
		//塗黑
		rectangle(matResult, rectIgnore, Scalar(-1), CV_FILLED);
		blockMax.UpdateMax(rectIgnore);
		Point ptReturn;
		blockMax.GetMaxValueLoc(dMaxValue, ptReturn);
		return ptReturn;
	}

	bool compareScoreBig2Small(const s_MatchParameter& lhs, const s_MatchParameter& rhs) { return  lhs.dMatchScore > rhs.dMatchScore; }
	bool comparePtWithAngle(const pair<Point2f, double> lhs, const pair<Point2f, double> rhs) { return lhs.second < rhs.second; }

	void GetRotatedROI(Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI)
	{
		double dAngle_radian = dAngle * D2R;
		Point2f ptC((matSrc.cols - 1) / 2.0f, (matSrc.rows - 1) / 2.0f);
		Point2f ptLT_rotate = ptRotatePt2f(ptLT, ptC, dAngle_radian);
		Size sizePadding(size.width + 6, size.height + 6);


		Mat rMat = getRotationMatrix2D(ptC, dAngle, 1);
		rMat.at<double>(0, 2) -= ptLT_rotate.x - 3;
		rMat.at<double>(1, 2) -= ptLT_rotate.y - 3;
		//平移旋轉矩陣(0, 2) (1, 2)的減，為旋轉後的圖形偏移，-= ptLT_rotate.x - 3 代表旋轉後的圖形往-X方向移動ptLT_rotate.x - 3
		//Debug

		//Debug
		warpAffine(matSrc, matROI, rMat, sizePadding);
	}

	bool SubPixEsimation(vector<s_MatchParameter>* vec, double* dNewX, double* dNewY, double* dNewAngle, double dAngleStep, int iMaxScoreIndex)
	{
		//Az=S, (A.T)Az=(A.T)s, z = ((A.T)A).inv (A.T)s

		Mat matA(27, 10, CV_64F);
		Mat matZ(10, 1, CV_64F);
		Mat matS(27, 1, CV_64F);

		double dX_maxScore = (*vec)[iMaxScoreIndex].pt.x;
		double dY_maxScore = (*vec)[iMaxScoreIndex].pt.y;
		double dTheata_maxScore = (*vec)[iMaxScoreIndex].dMatchAngle;
		int iRow = 0;
		/*for (int x = -1; x <= 1; x++)
		{
			for (int y = -1; y <= 1; y++)
			{
				for (int theta = 0; theta <= 2; theta++)
				{*/
		for (int theta = 0; theta <= 2; theta++)
		{
			for (int y = -1; y <= 1; y++)
			{
				for (int x = -1; x <= 1; x++)
				{
					//xx yy tt xy xt yt x y t 1
					//0  1  2  3  4  5  6 7 8 9
					double dX = dX_maxScore + x;
					double dY = dY_maxScore + y;
					//double dT = (*vec)[theta].dMatchAngle + (theta - 1) * dAngleStep;
					double dT = (dTheata_maxScore + (theta - 1) * dAngleStep) * D2R;
					matA.at<double>(iRow, 0) = dX * dX;
					matA.at<double>(iRow, 1) = dY * dY;
					matA.at<double>(iRow, 2) = dT * dT;
					matA.at<double>(iRow, 3) = dX * dY;
					matA.at<double>(iRow, 4) = dX * dT;
					matA.at<double>(iRow, 5) = dY * dT;
					matA.at<double>(iRow, 6) = dX;
					matA.at<double>(iRow, 7) = dY;
					matA.at<double>(iRow, 8) = dT;
					matA.at<double>(iRow, 9) = 1.0;
					matS.at<double>(iRow, 0) = (*vec)[iMaxScoreIndex + (theta - 1)].vecResult[x + 1][y + 1];
					iRow++;
#ifdef _DEBUG
					/*string str = format ("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f", dValueA[0], dValueA[1], dValueA[2], dValueA[3], dValueA[4], dValueA[5], dValueA[6], dValueA[7], dValueA[8], dValueA[9]);
					fileA <<  str << endl;
					str = format ("%.6f", dValueS[iRow]);
					fileS << str << endl;*/
#endif
				}
			}
		}
		//求解Z矩陣，得到k0~k9
		//[ x* ] = [ 2k0 k3 k4 ]-1 [ -k6 ]
		//| y* | = | k3 2k1 k5 |   | -k7 |
		//[ t* ] = [ k4 k5 2k2 ]   [ -k8 ]

		//solve (matA, matS, matZ, DECOMP_SVD);
		matZ = (matA.t() * matA).inv() * matA.t() * matS;
		Mat matZ_t;
		transpose(matZ, matZ_t);
		double* dZ = matZ_t.ptr<double>(0);
		Mat matK1 = (Mat_<double>(3, 3) <<
			(2 * dZ[0]), dZ[3], dZ[4],
			dZ[3], (2 * dZ[1]), dZ[5],
			dZ[4], dZ[5], (2 * dZ[2]));
		Mat matK2 = (Mat_<double>(3, 1) << -dZ[6], -dZ[7], -dZ[8]);
		Mat matDelta = matK1.inv() * matK2;

		*dNewX = matDelta.at<double>(0, 0);
		*dNewY = matDelta.at<double>(1, 0);
		*dNewAngle = matDelta.at<double>(2, 0) * R2D;
		return true;
	}

	void FilterWithScore(vector<s_MatchParameter>* vec, double dScore)
	{
		std::sort(vec->begin(), vec->end(), compareScoreBig2Small);
		int iSize = vec->size(), iIndexDelete = iSize + 1;
		for (int i = 0; i < iSize; i++)
		{
			if ((*vec)[i].dMatchScore < dScore)
			{
				iIndexDelete = i;
				break;
			}
		}
		if (iIndexDelete == iSize + 1)//沒有任何元素小於dScore
			return;
		vec->erase(vec->begin() + iIndexDelete, vec->end());
		return;
	}

	void SortPtWithCenter(vector<Point2f>& vecSort)
	{
		int iSize = (int)vecSort.size();
		Point2f ptCenter;
		for (int i = 0; i < iSize; i++)
			ptCenter += vecSort[i];
		ptCenter /= iSize;

		Point2f vecX(1, 0);

		vector<pair<Point2f, double>> vecPtAngle(iSize);
		for (int i = 0; i < iSize; i++)
		{
			vecPtAngle[i].first = vecSort[i];//pt
			Point2f vec1(vecSort[i].x - ptCenter.x, vecSort[i].y - ptCenter.y);
			float fNormVec1 = vec1.x * vec1.x + vec1.y * vec1.y;
			float fDot = vec1.x;

			if (vec1.y < 0)//若點在中心的上方
			{
				vecPtAngle[i].second = acos(fDot / fNormVec1) * R2D;
			}
			else if (vec1.y > 0)//下方
			{
				vecPtAngle[i].second = 360 - acos(fDot / fNormVec1) * R2D;
			}
			else//點與中心在相同Y
			{
				if (vec1.x - ptCenter.x > 0)
					vecPtAngle[i].second = 0;
				else
					vecPtAngle[i].second = 180;
			}

		}
		std::sort(vecPtAngle.begin(), vecPtAngle.end(), comparePtWithAngle);
		for (int i = 0; i < iSize; i++)
			vecSort[i] = vecPtAngle[i].first;
	}

	void FilterWithRotatedRect(vector<s_MatchParameter>* vec, int iMethod, double dMaxOverLap)
	{
		int iMatchSize = (int)vec->size();
		RotatedRect rect1, rect2;
		for (int i = 0; i < iMatchSize - 1; i++)
		{
			if (vec->at(i).bDelete)
				continue;
			for (int j = i + 1; j < iMatchSize; j++)
			{
				if (vec->at(j).bDelete)
					continue;
				rect1 = vec->at(i).rectR;
				rect2 = vec->at(j).rectR;
				vector<Point2f> vecInterSec;
				int iInterSecType = rotatedRectangleIntersection(rect1, rect2, vecInterSec);
				if (iInterSecType == INTERSECT_NONE)//無交集
					continue;
				else if (iInterSecType == INTERSECT_FULL) //一個矩形包覆另一個
				{
					int iDeleteIndex;
					if (iMethod == CV_TM_SQDIFF)
						iDeleteIndex = (vec->at(i).dMatchScore <= vec->at(j).dMatchScore) ? j : i;
					else
						iDeleteIndex = (vec->at(i).dMatchScore >= vec->at(j).dMatchScore) ? j : i;
					vec->at(iDeleteIndex).bDelete = true;
				}
				else//交點 > 0
				{
					if (vecInterSec.size() < 3)//一個或兩個交點
						continue;
					else
					{
						int iDeleteIndex;
						//求面積與交疊比例
						//SortPtWithCenter(vecInterSec);
						vector<cv::Point2f> order_pts;
						cv::convexHull(cv::Mat(vecInterSec), order_pts, true);
						double dArea = contourArea(order_pts);
						double dRatio = dArea / rect1.size.area();
						//若大於最大交疊比例，選分數高的
						if (dRatio > dMaxOverLap)
						{
							if (iMethod == CV_TM_SQDIFF)
								iDeleteIndex = (vec->at(i).dMatchScore <= vec->at(j).dMatchScore) ? j : i;
							else
								iDeleteIndex = (vec->at(i).dMatchScore >= vec->at(j).dMatchScore) ? j : i;
							vec->at(iDeleteIndex).bDelete = true;
						}
					}
				}
			}
		}
		vector<s_MatchParameter>::iterator it;
		for (it = vec->begin(); it != vec->end();)
		{
			if ((*it).bDelete)
				it = vec->erase(it);
			else
				++it;
		}
	}

	PatternMatcher::PatternMatcher(const MatcherParam& param)
	{
		initFinishedFlag_ = false;

		initFinishedFlag_ = initMatcher(param);
		if (initFinishedFlag_ == false)
		{
			logger_->info("Failed to init matcher!");
			return;
		}

		logger_->info("成功!");

		initFinishedFlag_ = true;
	}

	PatternMatcher::~PatternMatcher()
	{

	}

	int PatternMatcher::match(const cv::Mat& image, std::vector<MatchResult>& matchResults)
	{
		matchResults.clear();
		if (image.empty())
			return -1;
		if (templateImage_.empty() || templateImage_.channels() != 1)
		{
			logger_->error("image empty or channels != 1.");
			return -5;
		}
		if ((templateImage_.cols < image.cols && templateImage_.rows > image.rows) || (templateImage_.cols > image.cols && templateImage_.rows < image.rows))
			return -2;
		if (templateImage_.size().area() > image.size().area())
			return -3;
		if (!m_TemplData.bIsPatternLearned)
			return -4;

		double d1 = clock();
		//決定金字塔層數 總共為1 + iLayer層
		int iTopLayer = GetTopLayer(&templateImage_, (int)sqrt((double)matchParam_.minArea));
		//建立金字塔
		vector<Mat> vecMatSrcPyr;
		buildPyramid(image, vecMatSrcPyr, iTopLayer);

		s_TemplData* pTemplData = &m_TemplData;

		//第一階段以最頂層找出大致角度與ROI
		double dAngleStep = atan(2.0 / max(pTemplData->vecPyramid[iTopLayer].cols, pTemplData->vecPyramid[iTopLayer].rows)) * R2D;
		//double dAngleStep = atan(2.0 / max(pTemplData->vecPyramid[0].cols, pTemplData->vecPyramid[0].rows)) * R2D;

		vector<double> vecAngles;

		if (matchParam_.angle < VISION_TOLERANCE)
			vecAngles.push_back(0.0);
		else
		{
			for (double dAngle = 0; dAngle < matchParam_.angle + dAngleStep; dAngle += dAngleStep)
				vecAngles.push_back(dAngle);
			for (double dAngle = -dAngleStep; dAngle > -matchParam_.angle - dAngleStep; dAngle -= dAngleStep)
				vecAngles.push_back(dAngle);
		}


		int iTopSrcW = vecMatSrcPyr[iTopLayer].cols, iTopSrcH = vecMatSrcPyr[iTopLayer].rows;
		Point2f ptCenter((iTopSrcW - 1) / 2.0f, (iTopSrcH - 1) / 2.0f);

		int iSize = (int)vecAngles.size();
		//vector<s_MatchParameter> vecMatchParameter (iSize * (m_iMaxPos + MATCH_CANDIDATE_NUM));
		vector<s_MatchParameter> vecMatchParameter;
		//Caculate lowest score at every layer
		vector<double> vecLayerScore(iTopLayer + 1, matchParam_.scoreThreshold);
		for (int iLayer = 1; iLayer <= iTopLayer; iLayer++)
			vecLayerScore[iLayer] = vecLayerScore[iLayer - 1] * 0.9;

		Size sizePat = pTemplData->vecPyramid[iTopLayer].size();
		bool bCalMaxByBlock = (vecMatSrcPyr[iTopLayer].size().area() / sizePat.area() > 500) && matchParam_.maxCount > 10;
		for (int i = 0; i < iSize; i++)
		{
			Mat matRotatedSrc, matR = getRotationMatrix2D(ptCenter, vecAngles[i], 1);
			Mat matResult;
			Point ptMaxLoc;
			double dValue, dMaxVal;
			double dRotate = clock();
			Size sizeBest = GetBestRotationSize(vecMatSrcPyr[iTopLayer].size(), pTemplData->vecPyramid[iTopLayer].size(), vecAngles[i]);

			float fTranslationX = (sizeBest.width - 1) / 2.0f - ptCenter.x;
			float fTranslationY = (sizeBest.height - 1) / 2.0f - ptCenter.y;
			matR.at<double>(0, 2) += fTranslationX;
			matR.at<double>(1, 2) += fTranslationY;
			warpAffine(vecMatSrcPyr[iTopLayer], matRotatedSrc, matR, sizeBest, INTER_LINEAR, BORDER_CONSTANT, Scalar(pTemplData->iBorderColor));

			MatchTemplate(matRotatedSrc, pTemplData, matResult, iTopLayer, false);

			if (bCalMaxByBlock)
			{
				s_BlockMax blockMax(matResult, pTemplData->vecPyramid[iTopLayer].size());
				blockMax.GetMaxValueLoc(dMaxVal, ptMaxLoc);
				if (dMaxVal < vecLayerScore[iTopLayer])
					continue;
				vecMatchParameter.push_back(s_MatchParameter(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngles[i]));
				for (int j = 0; j < matchParam_.maxCount + MATCH_CANDIDATE_NUM - 1; j++)
				{
					ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, pTemplData->vecPyramid[iTopLayer].size(), dValue, matchParam_.iouThreshold, blockMax);
					if (dValue < vecLayerScore[iTopLayer])
						break;
					vecMatchParameter.push_back(s_MatchParameter(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngles[i]));
				}
			}
			else
			{
				minMaxLoc(matResult, 0, &dMaxVal, 0, &ptMaxLoc);
				if (dMaxVal < vecLayerScore[iTopLayer])
					continue;
				vecMatchParameter.push_back(s_MatchParameter(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngles[i]));
				for (int j = 0; j < matchParam_.maxCount + MATCH_CANDIDATE_NUM - 1; j++)
				{
					ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, pTemplData->vecPyramid[iTopLayer].size(), dValue, matchParam_.iouThreshold);
					if (dValue < vecLayerScore[iTopLayer])
						break;
					vecMatchParameter.push_back(s_MatchParameter(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngles[i]));
				}
			}
		}
		std::sort(vecMatchParameter.begin(), vecMatchParameter.end(), compareScoreBig2Small);


		int iMatchSize = (int)vecMatchParameter.size();
		int iDstW = pTemplData->vecPyramid[iTopLayer].cols, iDstH = pTemplData->vecPyramid[iTopLayer].rows;

		//顯示第一層結果
		if (m_bDebugMode)
		{
			int iDebugScale = 2;

			Mat matShow, matResize;
			resize(vecMatSrcPyr[iTopLayer], matResize, vecMatSrcPyr[iTopLayer].size() * iDebugScale);
			cvtColor(matResize, matShow, CV_GRAY2BGR);
			string str = format("Toplayer, Candidate:%d", iMatchSize);
			vector<Point2f> vec;
			for (int i = 0; i < iMatchSize; i++)
			{
				Point2f ptLT, ptRT, ptRB, ptLB;
				double dRAngle = -vecMatchParameter[i].dMatchAngle * D2R;
				ptLT = ptRotatePt2f(vecMatchParameter[i].pt, ptCenter, dRAngle);
				ptRT = Point2f(ptLT.x + iDstW * (float)cos(dRAngle), ptLT.y - iDstW * (float)sin(dRAngle));
				ptLB = Point2f(ptLT.x + iDstH * (float)sin(dRAngle), ptLT.y + iDstH * (float)cos(dRAngle));
				ptRB = Point2f(ptRT.x + iDstH * (float)sin(dRAngle), ptRT.y + iDstH * (float)cos(dRAngle));
				line(matShow, ptLT * iDebugScale, ptLB * iDebugScale, Scalar(0, 255, 0));
				line(matShow, ptLB * iDebugScale, ptRB * iDebugScale, Scalar(0, 255, 0));
				line(matShow, ptRB * iDebugScale, ptRT * iDebugScale, Scalar(0, 255, 0));
				line(matShow, ptRT * iDebugScale, ptLT * iDebugScale, Scalar(0, 255, 0));
				circle(matShow, ptLT * iDebugScale, 1, Scalar(0, 0, 255));
				vec.push_back(ptLT * iDebugScale);
				vec.push_back(ptRT * iDebugScale);
				vec.push_back(ptLB * iDebugScale);
				vec.push_back(ptRB * iDebugScale);

				string strText = format("%d", i);
				cv::putText(matShow, strText, ptLT * iDebugScale, FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
			}
			cvNamedWindow(str.c_str(), 0x10000000);
			Rect rectShow = boundingRect(vec);
			cv::imshow(str, matShow);// (rectShow));
			cv::waitKey();
			//moveWindow (str, 0, 0);
		}
		//顯示第一層結果

		//第一階段結束
		bool bSubPixelEstimation = m_bSubPixel;
		int iStopLayer = m_bStopLayer1 ? 1 : 0; //设置为1时：粗匹配，牺牲精度提升速度。
		//int iSearchSize = min (m_iMaxPos + MATCH_CANDIDATE_NUM, (int)vecMatchParameter.size ());//可能不需要搜尋到全部 太浪費時間
		vector<s_MatchParameter> vecAllResult;
		for (int i = 0; i < (int)vecMatchParameter.size(); i++)
			//for (int i = 0; i < iSearchSize; i++)
		{
			double dRAngle = -vecMatchParameter[i].dMatchAngle * D2R;
			Point2f ptLT = ptRotatePt2f(vecMatchParameter[i].pt, ptCenter, dRAngle);

			double dAngleStep = atan(2.0 / max(iDstW, iDstH)) * R2D;//min改為max
			vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep;
			vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep;

			if (iTopLayer <= iStopLayer)
			{
				vecMatchParameter[i].pt = Point2d(ptLT * ((iTopLayer == 0) ? 1 : 2));
				vecAllResult.push_back(vecMatchParameter[i]);
			}
			else
			{
				for (int iLayer = iTopLayer - 1; iLayer >= iStopLayer; iLayer--)
				{
					//搜尋角度
					dAngleStep = atan(2.0 / max(pTemplData->vecPyramid[iLayer].cols, pTemplData->vecPyramid[iLayer].rows)) * R2D;//min改為max
					vector<double> vecAngles;
					//double dAngleS = vecMatchParameter[i].dAngleStart, dAngleE = vecMatchParameter[i].dAngleEnd;
					double dMatchedAngle = vecMatchParameter[i].dMatchAngle;
					if (matchParam_.angle)
					{
						for (int i = -2; i <= 2; i++)
							vecAngles.push_back(dMatchedAngle + dAngleStep * i);
					}
					else
					{
						if (matchParam_.angle < VISION_TOLERANCE)
							vecAngles.push_back(0.0);
						else
							for (int i = -2; i <= 2; i++)
								vecAngles.push_back(dMatchedAngle + dAngleStep * i);
					}
					Point2f ptSrcCenter((vecMatSrcPyr[iLayer].cols - 1) / 2.0f, (vecMatSrcPyr[iLayer].rows - 1) / 2.0f);
					iSize = (int)vecAngles.size();
					vector<s_MatchParameter> vecNewMatchParameter(iSize);
					int iMaxScoreIndex = 0;
					double dBigValue = -1;
					for (int j = 0; j < iSize; j++)
					{
						Mat matResult, matRotatedSrc;
						double dMaxValue = 0;
						Point ptMaxLoc;
						GetRotatedROI(vecMatSrcPyr[iLayer], pTemplData->vecPyramid[iLayer].size(), ptLT * 2, vecAngles[j], matRotatedSrc);

						MatchTemplate(matRotatedSrc, pTemplData, matResult, iLayer, true);
						//matchTemplate (matRotatedSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCOEFF_NORMED);
						minMaxLoc(matResult, 0, &dMaxValue, 0, &ptMaxLoc);
						vecNewMatchParameter[j] = s_MatchParameter(ptMaxLoc, dMaxValue, vecAngles[j]);

						if (vecNewMatchParameter[j].dMatchScore > dBigValue)
						{
							iMaxScoreIndex = j;
							dBigValue = vecNewMatchParameter[j].dMatchScore;
						}
						//次像素估計
						if (ptMaxLoc.x == 0 || ptMaxLoc.y == 0 || ptMaxLoc.x == matResult.cols - 1 || ptMaxLoc.y == matResult.rows - 1)
							vecNewMatchParameter[j].bPosOnBorder = true;
						if (!vecNewMatchParameter[j].bPosOnBorder)
						{
							for (int y = -1; y <= 1; y++)
								for (int x = -1; x <= 1; x++)
									vecNewMatchParameter[j].vecResult[x + 1][y + 1] = matResult.at<float>(ptMaxLoc + Point(x, y));
						}
						//次像素估計
					}
					if (vecNewMatchParameter[iMaxScoreIndex].dMatchScore < vecLayerScore[iLayer])
						break;
					//次像素估計
					if (bSubPixelEstimation
						&& iLayer == 0
						&& (!vecNewMatchParameter[iMaxScoreIndex].bPosOnBorder)
						&& iMaxScoreIndex != 0
						&& iMaxScoreIndex != 2)
					{
						double dNewX = 0, dNewY = 0, dNewAngle = 0;
						SubPixEsimation(&vecNewMatchParameter, &dNewX, &dNewY, &dNewAngle, dAngleStep, iMaxScoreIndex);
						vecNewMatchParameter[iMaxScoreIndex].pt = Point2d(dNewX, dNewY);
						vecNewMatchParameter[iMaxScoreIndex].dMatchAngle = dNewAngle;
					}
					//次像素估計

					double dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex].dMatchAngle;

					//讓坐標系回到旋轉時(GetRotatedROI)的(0, 0)
					Point2f ptPaddingLT = ptRotatePt2f(ptLT * 2, ptSrcCenter, dNewMatchAngle * D2R) - Point2f(3, 3);
					Point2f pt(vecNewMatchParameter[iMaxScoreIndex].pt.x + ptPaddingLT.x, vecNewMatchParameter[iMaxScoreIndex].pt.y + ptPaddingLT.y);
					//再旋轉
					pt = ptRotatePt2f(pt, ptSrcCenter, -dNewMatchAngle * D2R);

					if (iLayer == iStopLayer)
					{
						vecNewMatchParameter[iMaxScoreIndex].pt = pt * (iStopLayer == 0 ? 1 : 2);
						vecAllResult.push_back(vecNewMatchParameter[iMaxScoreIndex]);
					}
					else
					{
						//更新MatchAngle ptLT
						vecMatchParameter[i].dMatchAngle = dNewMatchAngle;
						vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep / 2;
						vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep / 2;
						ptLT = pt;
					}
				}

			}
		}
		FilterWithScore(&vecAllResult, matchParam_.scoreThreshold);

		//最後濾掉重疊
		iDstW = pTemplData->vecPyramid[iStopLayer].cols * (iStopLayer == 0 ? 1 : 2);
		iDstH = pTemplData->vecPyramid[iStopLayer].rows * (iStopLayer == 0 ? 1 : 2);

		for (int i = 0; i < (int)vecAllResult.size(); i++)
		{
			Point2f ptLT, ptRT, ptRB, ptLB;
			double dRAngle = -vecAllResult[i].dMatchAngle * D2R;
			ptLT = vecAllResult[i].pt;
			ptRT = Point2f(ptLT.x + iDstW * (float)cos(dRAngle), ptLT.y - iDstW * (float)sin(dRAngle));
			ptLB = Point2f(ptLT.x + iDstH * (float)sin(dRAngle), ptLT.y + iDstH * (float)cos(dRAngle));
			ptRB = Point2f(ptRT.x + iDstH * (float)sin(dRAngle), ptRT.y + iDstH * (float)cos(dRAngle));
			//紀錄旋轉矩形
			vecAllResult[i].rectR = RotatedRect(ptLT, ptRT, ptRB);
		}
		FilterWithRotatedRect(&vecAllResult, CV_TM_CCOEFF_NORMED, matchParam_.iouThreshold);
		//最後濾掉重疊

		//根據分數排序
		std::sort(vecAllResult.begin(), vecAllResult.end(), compareScoreBig2Small);

		iMatchSize = (int)vecAllResult.size();
		if (vecAllResult.size() == 0)
			return false;
		int iW = pTemplData->vecPyramid[0].cols, iH = pTemplData->vecPyramid[0].rows;

		for (int i = 0; i < iMatchSize; i++)
		{
			MatchResult sstm;
			double dRAngle = -vecAllResult[i].dMatchAngle * D2R;

			sstm.LeftTop = vecAllResult[i].pt;

			sstm.RightTop = Point2d(sstm.LeftTop.x + iW * cos(dRAngle), sstm.LeftTop.y - iW * sin(dRAngle));
			sstm.LeftBottom = Point2d(sstm.LeftTop.x + iH * sin(dRAngle), sstm.LeftTop.y + iH * cos(dRAngle));
			sstm.RightBottom = Point2d(sstm.RightTop.x + iH * sin(dRAngle), sstm.RightTop.y + iH * cos(dRAngle));
			sstm.Center = Point2d((sstm.LeftTop.x + sstm.RightTop.x + sstm.RightBottom.x + sstm.LeftBottom.x) / 4, (sstm.LeftTop.y + sstm.RightTop.y + sstm.RightBottom.y + sstm.LeftBottom.y) / 4);
			sstm.Angle = -vecAllResult[i].dMatchAngle;
			sstm.Score = vecAllResult[i].dMatchScore;

			if (sstm.Angle < -180)
				sstm.Angle += 360;
			if (sstm.Angle > 180)
				sstm.Angle -= 360;
			matchResults.push_back(sstm);

			//std::cout << "x: " << sstm.Center.x << ", y: " << sstm.Center.y << endl;



			//Test Subpixel
			/*Point2d ptLT = vecAllResult[i].ptSubPixel;
			Point2d ptRT = Point2d (sstm.ptLT.x + iW * cos (dRAngle), sstm.ptLT.y - iW * sin (dRAngle));
			Point2d ptLB = Point2d (sstm.ptLT.x + iH * sin (dRAngle), sstm.ptLT.y + iH * cos (dRAngle));
			Point2d ptRB = Point2d (sstm.ptRT.x + iH * sin (dRAngle), sstm.ptRT.y + iH * cos (dRAngle));
			Point2d ptCenter = Point2d ((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4, (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
			CString strDiff;strDiff.Format (L"Diff(x, y):%.3f, %.3f", ptCenter.x - sstm.ptCenter.x, ptCenter.y - sstm.ptCenter.y);
			AfxMessageBox (strDiff);*/
			//Test Subpixel
			if (i + 1 == matchParam_.maxCount)
				break;
		}

		return (int)matchResults.size();
	}

	int PatternMatcher::setTemplate(const cv::Mat& templateImage)
	{
		if (templateImage.empty())
			return -1;
		if (templateImage.channels() > 1)
			return -2;
		m_TemplData.bIsPatternLearned = false;

		templateImage_ = templateImage.clone();
		LearnPattern(m_TemplData, templateImage_, matchParam_.minArea);

		return 0;

	}

}
