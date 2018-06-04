#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc.hpp"	// resize()

using namespace cv;

static void help()
{
    printf("\nThis program demonstrates using features2d detector, descriptor extractor and simple matcher\n"
            "Using the SIFT desriptor:\n"
            "\n"
            "Usage:\n matcher <input_image1> <input_image2> <output>\n");
}

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        help();
        return -1;
    }

    Mat ori_img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat ori_img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if(ori_img1.empty() || ori_img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }

	// resize images
	Mat img1, img2;
    // resize(ori_img1, img1, Size(), 0.25, 0.25, INTER_LINEAR);
    // resize(ori_img2, img2, Size(), 0.25, 0.25, INTER_LINEAR);

    resize(ori_img1, img1, Size(), 1, 1, INTER_LINEAR);
    resize(ori_img2, img2, Size(), 1, 1, INTER_LINEAR);

    // detecting keypoints
    SiftFeatureDetector detector( 2000);
    vector<KeyPoint> keypoints1, keypoints2;
    detector.detect(img1, keypoints1);
    detector.detect(img2, keypoints2);

    // computing descriptors
    SiftDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);

    // matching descriptors
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    double max_dist = 0; 
    double min_dist = 500;
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors1.rows; i++ )
    { 
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors1.rows; i++ )
    { 
        // if( matches[i].distance <= max(2*min_dist, 0.02) )
        if( matches[i].distance <= max(5*min_dist, 0.1) )
        { 
            good_matches.push_back( matches[i]); 
        }
    }


    // drawing the results
    namedWindow("matches", 1);
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches);
    namedWindow("matches", WINDOW_NORMAL);
	imshow("matches", img_matches);

    for(int i=0; i< (int)good_matches.size(); ++i)
    {
        printf("Good Match [%d] Keypoint1: %d --- Keypoint2: %d \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
    }

    Mat out_img;
    imwrite( argv[3], img_matches);

    waitKey(0);

    return 0;
}
