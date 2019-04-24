#pragma once

#include <ros/ros.h>

#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>


#include "/home/lab606/catkin_vins/src/VINS-Mono/pose_graph/src/ThirdParty/DVision/DVision.h"
//#include "/home/lab606/catkin_vins/src/VINS-Mono/pose_graph/src/ThirdParty/DBoW/TemplatedDatabase.h"
#include "/home/lab606/catkin_vins/src/VINS-Mono/pose_graph/src/ThirdParty/DBoW/DBoW2.h"
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include "utility/tic_toc.h"
#include <string>
using namespace std;
using namespace DVision;
using namespace Eigen;
using namespace DBoW2;

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string IMU_TOPIC;
extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern double ROW, COL;


void readParameters(ros::NodeHandle &n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};


class initKeyFrame
{
public:

	initKeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);

	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);

	double time_stamp; 
	int index;
	Eigen::Vector3d vio_T_w_i; 
	Eigen::Matrix3d vio_R_w_i; 
	Eigen::Vector3d T_w_i;
	Eigen::Matrix3d R_w_i;
	Eigen::Vector3d origin_vio_T;		
	Eigen::Matrix3d origin_vio_R;
	cv::Mat image;
	vector<cv::Point3f> point_3d; 
	vector<cv::Point2f> point_2d_uv;
	vector<cv::Point2f> point_2d_norm;
	vector<double> point_id;
	vector<cv::KeyPoint> keypoints;
	vector<cv::KeyPoint> keypoints_norm;
	vector<BRIEF::bitset> brief_descriptors;
	vector<BRIEF::bitset> window_brief_descriptors;

	int sequence;

	int loop_index;
	Eigen::Matrix<double, 8, 1 > loop_info;
};
//BriefDatabase db;

void saveMapToInit();//载入历史地图用于初始化
void loadKeyFrame(initKeyFrame* kf);
void addKeyFrameIntoVoc(initKeyFrame* keyframe);
