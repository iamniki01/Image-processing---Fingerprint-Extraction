#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using  namespace cv;

//=======class declaration and definiton for skin color binarization========//
class skin_binarization
{

	////data members
public:
	cv::Mat frame;
	Mat FrameHSV, FingerSegmentation;
	Mat src_hsv, skin_mask;	
	
	

private:
	int iLowH;
	int iHighH;

	int iLowS;
	int iHighS;

	int iLowB ;
	int iHighB;


public:

	////=======constructor declaring maximum and minimum values for HSV ==============/////
	///member function 1
	///===constructor=======//
	skin_binarization::skin_binarization(void)
	{
		 iLowH = 0;
		 iHighH = 120;

		 iLowS = 58;
		 iHighS = 180;

		 iLowB = 50;
		 iHighB = 245;

	}

	///member function 2

	///========= Drawing aline from the end to extract only fingers======================///

	void MyLine(Mat img, Point start, Point end)
	{
		int thickness = 550;
		int lineType = LINE_8;
		line(img, start, end, Scalar(0, 0, 255), thickness, lineType);
	}

	///member function 3
	///=========================This functtion is to provide good range for different brightness level ==============///
	////======not been used in this , since the demo is done is closed room with white background so no range is required======//

	Mat hist_eq(Mat src)
	{
		Mat chnl[3], dst, red, green, blue;

		split(src, chnl);
		vector <Mat> channels = { chnl[0], chnl[1], chnl[2] };

		equalizeHist(channels[0], blue);
		equalizeHist(channels[1], green);
		equalizeHist(channels[2], red);
		vector <Mat> eq_channels = { blue, green, red };
		merge(eq_channels, dst);

		return(dst);
	}
	///member function 
	///=========================This functtion is to provide trackbbar to adjust different HSV level ==============///

	void on_trackbar()
	{
		int desiredWidth = 320, desiredheight = 380;
		namedWindow("FrameHSV", CV_WINDOW_NORMAL);

		resizeWindow("FrameHSV", desiredWidth, desiredheight);

		//namedWindow("FrameHSV", CV_WINDOW_NORMAL);
		createTrackbar("LowHue", "FrameHSV", &iLowH, 180);
		createTrackbar("HighHue", "FrameHSV", &iHighH, 255);

		createTrackbar("LowSat", "FrameHSV", &iLowS, 255);
		createTrackbar("HighSat", "FrameHSV", &iHighS, 255);

		createTrackbar("LowBright", "FrameHSV", &iLowB, 255);
		createTrackbar("HighBright", "FrameHSV", &iHighB, 255);
	}



	///member function 4

	Mat skin_binarization::fingerSegmentation(cv::Mat input)
	{

		Mat img = input.clone();
		int elementSize = 2;

		//FrameHSV = hist_eq(img);

		//======calling function for trackbar==========///
		on_trackbar();

		cvtColor(img, FrameHSV, CV_BGR2HSV);  // converting to BGR to HSV 

		///============== appplying filter to reduce noise before thresholding ==============///
		medianBlur(FrameHSV, FrameHSV,5);
		blur(FrameHSV, FrameHSV, Size(15, 15));

		//=========== binirizing by using inrange function ===================// 

		inRange(FrameHSV, Scalar(iLowH, iLowS, iLowB), Scalar(iHighH, iHighS, iHighB), skin_mask);
		blur(skin_mask, skin_mask, Size(5, 5));
		imshow("binary1", skin_mask);

		///================= Morphological transform using operations like Erosion, Dilation=========================///

		cv::Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(elementSize * 2 + 1, elementSize * 2 + 1), Point(elementSize, elementSize));
		MyLine(skin_mask, { 0, input.rows }, { input.cols, input.rows });
		cv::erode(skin_mask, FingerSegmentation, element);
		cv::dilate(skin_mask, FingerSegmentation, element);
		//imshow("Mask", FingerSegmentation);

		///============= to find Contours using minArearect and adjusting center==========////


		vector<vector<cv::Point>> Contours;
		vector<Vec4i>heirarchy;
		int largestContour = 0;
		findContours(FingerSegmentation, Contours, heirarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		int height = 0;
		int width = 0;
		vector<RotatedRect> minRect(Contours.size());
		for (size_t i = 0; i < Contours.size(); i++)
		{
		
		minRect[i] = minAreaRect(Mat(Contours[i]));


		// Convex hull to dectect fingertips
		
			if (!Contours.empty())
			{
				std::vector<std::vector<cv::Point> > hull(1);
				cv::convexHull(cv::Mat(Contours[i]), hull[0], false);
				cv::drawContours(input, hull, 0, cv::Scalar(0, 255, 0), 1); //green and thickness =1
				if (hull[0].size() > 5)
				{
					std::vector<int> hullIndexes;
					cv::convexHull(cv::Mat(Contours[i]), hullIndexes, true);
					std::vector<cv::Vec4i> convexityDefects;
					cv::convexityDefects(cv::Mat(Contours[i]), hullIndexes, convexityDefects);
				}
			}
		}			

		for (size_t i = 0; i < Contours.size(); i++)
		{
			if (Contours[i].size() > 150)
			{
				//we only want to keep big contour

				Scalar color = Scalar(255, 0, 0);
				height = minRect[i].size.height;
				width = minRect[i].size.width;
				Point2f rect_points[4];
				minRect[i].points(rect_points);


				//drawing rectangle
				if (height > width)
				{
					//====adjusting height and center for extract finger tip============//  
					minRect[i].size.height = (float)(0.43)*minRect[i].size.height;
					minRect[i].center = (rect_points[1] + rect_points[2]) / 2 + (rect_points[0] - rect_points[1]) / 6;
				}
				else
				{
					//====adjusting width and center for extract finger tip============// 
					minRect[i].size.width = (float)(0.43)*minRect[i].size.width;
					minRect[i].center = (rect_points[2] + rect_points[3]) / 2 + (rect_points[0] - rect_points[3]) / 6;
				}
				minRect[i].points(rect_points);

				//====adjusting width, heightand  center after extraction finger tip and drawing rectangle ============// 
				for (int j = 0; j < 4; j++)
					line(input, rect_points[j], rect_points[(j + 1) % 4], color, 2, 8);
			}
		}
		
		return(input);
	}
};



int main()
{
	Mat frame, cameraFeed;
	Mat skinMat;
	cv::VideoCapture cap(0); // 0: opens the default camera, 1: other
	if (!cap.isOpened())		// check if we succeeded
	{
		cout << "camera error" << endl;
		return -1;
	}




	// Retrieve frame size
	int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	// Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
	cv::VideoWriter video("out.avi",
		CV_FOURCC('M', 'J', 'P', 'G'),
		25, // fps
		cv::Size(frame_width, frame_height));
	

	skin_binarization  mySkinDetector;



	while (1){


		cap.read(cameraFeed);

		skinMat = mySkinDetector.fingerSegmentation(cameraFeed);

		imshow("FINAL VIDEO", skinMat);
		if (cv::waitKey(1) >= 0) break;

	}
	cap.release();
	cap.release();
	return 0;
}
