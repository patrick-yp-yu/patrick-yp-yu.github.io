// Project: Correct image perspective
// Yuan-Peng Yu
//
// removing perspective distortion by 4 points

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // read image
    Mat img1st = imread("IMG_book.jpg", CV_LOAD_IMAGE_COLOR);

    // the coordinate in the first view. Give 4 points.
    double x_1st[4];	// x
    double y_1st[4];	// y
    // the coordinate in the second view. Give 4points.
    double x_2nd[4];	// x'
    double y_2nd[4];	// y'

    vector<Point2f> pts_src;	//source points
    vector<Point2f> pts_dst;	//destination points

    // Book_Multiview Geometry (24.5cm  x 17.5cm)
    // input set4 (clockwised)
    x_1st[0] = 1921; y_1st[0] = 194;	// Point0
   	x_1st[1] = 3147; y_1st[1] = 368;	// Point1
   	x_1st[2] = 1686; y_1st[2] = 2624;	// Point2
   	x_1st[3] = 723; y_1st[3] = 1807;	// Point3
    pts_src.push_back(Point2f(1921, 194));
    pts_src.push_back(Point2f(3147, 368));
    pts_src.push_back(Point2f(1686, 2624));
    pts_src.push_back(Point2f(723, 1807));

    // Selected points in the second view.
    x_2nd[0] = 0; y_2nd[0] = 0;
   	x_2nd[1] = 350; y_2nd[1] = 0;
   	x_2nd[2] = 350; y_2nd[2] = 490;
   	x_2nd[3] = 0; y_2nd[3] = 490;
    pts_dst.push_back(Point2f(0, 0));
    pts_dst.push_back(Point2f(350, 0));
    pts_dst.push_back(Point2f(350, 490));
    pts_dst.push_back(Point2f(0, 490));

    // Use for verification
    Mat h_verify = findHomography(pts_src, pts_dst);

    // Start of the DLT algorithm
    Mat A(8,9, CV_32F);	// 8x9 matrix
    Mat H_vector(9,1, CV_32F);	// 9x1 matrix

    for (int p=0; p<4; ++p){	// 4 points
			A.at<float>(2*p,0) = x_1st[p];	// x
			A.at<float>(2*p,1) = y_1st[p];	// y
			A.at<float>(2*p,2) = 1;
			A.at<float>(2*p,3) = 0;
			A.at<float>(2*p,4) = 0;
			A.at<float>(2*p,5) = 0;
			A.at<float>(2*p,6) = -( x_1st[p] * x_2nd[p] );		// -xx'
			A.at<float>(2*p,7) = -( y_1st[p] * x_2nd[p] ); ; 	// -yx'
			A.at<float>(2*p,8) = -x_2nd[p];

			A.at<float>(2*p+1,0) = 0;	// x
			A.at<float>(2*p+1,1) = 0;	// y
			A.at<float>(2*p+1,2) = 0;
			A.at<float>(2*p+1,3) = x_1st[p];
			A.at<float>(2*p+1,4) = y_1st[p];
			A.at<float>(2*p+1,5) = 1;
			A.at<float>(2*p+1,6) = -( x_1st[p] * y_2nd[p] );	// -xy'
			A.at<float>(2*p+1,7) = -( y_1st[p] * y_2nd[p] );  	// -yy'
			A.at<float>(2*p+1,8) = -y_2nd[p];	//
    }

	cout << "A = " << endl << A << endl << endl;

	// use SVD::compute to gain homography matrix H
	Mat w, u, vt, V, out;
	SVD::compute(A, w, u, vt, SVD::FULL_UV);
	V = vt.t();
	H_vector = V.col(8);	// the last column = H
	cout << "H = " << endl << H_vector << endl << endl;

	out = A * H_vector;
	cout << "Verify A*H = 0 " << endl << out << endl << endl;

	Mat H_matrix(3,3, CV_32F);	// 3x3 matrix
	int k=0;
	for(int i=0; i<3; ++i){
		for(int j=0; j<3; ++j){
			H_matrix.at<float>(i,j) = H_vector.at<float>(k,0);
			++k;
		}
	}
	cout << "Homography matrix H = " << endl << H_matrix << endl << endl;

	// Show selected points
	for(int i=0; i<4; ++i){
	 	circle( img1st, Point( x_1st[i], y_1st[i] ), 20, Scalar( 0, 255, 255), 10, 8 );
	}

    Size size2(350,490);

	namedWindow("Original image", WINDOW_NORMAL);
    imshow("Original image", img1st);

	Mat img2nd;
	warpPerspective(img1st, img2nd, H_matrix, size2 );
	namedWindow("Image rectified by 4-points method", WINDOW_NORMAL);
    imshow("Image rectified by 4-points method", img2nd);

	// verify H by findHomography
	Mat img3rd;
	warpPerspective(img1st, img3rd, h_verify, size2 );
	cout << "Verify by findHomography = " << endl << h_verify << endl << endl;

	namedWindow("Image rectified by findHomography()", WINDOW_NORMAL);
    imshow("Image rectified by findHomography()", img3rd);
    // verify end

    imwrite( "book_Original.jpg", img1st );
    imwrite( "book_Rectified.jpg", img2nd );

    waitKey(0);
    return 0;
}
