#include "keyframe.h"

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// create keyframe online创建新的关键帧
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
		           vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
		           vector<double> &_point_id, int _sequence)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;		
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	computeWindowBRIEFPoint();//计算窗口中现有所有特帧点的描述子
	computeBRIEFPoint();//额外检测新的特征点并计算所有特征点的描述子
	if(!DEBUG_IMAGE)
		image.release();
}

// load previous keyframe载入原始关键帧
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
					cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
					vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors)
{
	time_stamp = _time_stamp;
	index = _index;
	//vio_T_w_i = _vio_T_w_i;
	//vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
	if (DEBUG_IMAGE)
	{
		image = _image.clone();
		cv::resize(image, thumbnail, cv::Size(80, 60));
	}
	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	sequence = 0;
	keypoints = _keypoints;
	keypoints_norm = _keypoints_norm;
	brief_descriptors = _brief_descriptors;
}


void KeyFrame::computeWindowBRIEFPoint()//计算窗口中现有所有特帧点的描述子
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	for(int i = 0; i < (int)point_2d_uv.size(); i++)
	{
	    cv::KeyPoint key;
	    key.pt = point_2d_uv[i];
	    window_keypoints.push_back(key);
	}
	extractor(image, window_keypoints, window_brief_descriptors);
}

void KeyFrame::computeBRIEFPoint()//额外检测新的特征点并计算所有特征点的描述子
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int fast_th = 20; // corner detector response threshold
	if(1)
		cv::FAST(image, keypoints, fast_th, true);
	else
	{
		vector<cv::Point2f> tmp_pts;
		cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);//输入的图像，保存检测到的角点，角点的最大个数，焦点的品质
		for(int i = 0; i < (int)tmp_pts.size(); i++)
		{
		    cv::KeyPoint key;
		    key.pt = tmp_pts[i];
		    keypoints.push_back(key);
		}
	}
	extractor(image, keypoints, brief_descriptors);
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}

void BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  m_brief.compute(im, keys, descriptors);
}


bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {

        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if(dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    //printf("best dist %d", bestDist);
    if (bestIndex != -1 && bestDist < 80)
    {
      best_match = keypoints_old[bestIndex].pt;
      best_match_norm = keypoints_old_norm[bestIndex].pt;
      return true;
    }
    else
      return false;
}
//searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
//将关键帧与回环帧进行BRIEF描述子匹配
void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
								std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
    for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
          status.push_back(1);
        else
          status.push_back(0);
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }

}

//通过RANSAC基本矩阵检验去除异常匹配点
void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status)
{
	int n = (int)matched_2d_cur_norm.size();
	for (int i = 0; i < n; i++)
		status.push_back(0);
    if (n >= 8)
    {
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
        {
            double FOCAL_LENGTH = 460.0;
            double tmp_x, tmp_y;
            tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
        cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}
//通过RANSAC的PnP检验去除异常匹配点
void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
	//for (int i = 0; i < matched_3d.size(); i++)
	//	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
	//printf("match size %d \n", matched_3d.size());
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;
    Matrix3d R_w_c = origin_vio_R * qic;
    Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;

    if (CV_MAJOR_VERSION < 3)
        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    else
    {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);

    }
	//solverPnPRansac
    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic.transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic;

}



bool KeyFrame::drawConnection(KeyFrame* old_kf)//为了计算相对位姿，最主要的就是利用pnpRansac函数，负责将匹配好的点发布到vins
{

	TicToc tmp_t;
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<uchar> status;
	//当前帧的数据
	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	TicToc t_match;
	//历史帧的数据
	searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);	
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	status.clear();
	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)//25	
	{
		status.clear();
		cout<<index<<"  :   "<<old_kf->index<<endl;
		//执行RANSAC处理之前
		int gap = 10;
		cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
		cv::Mat gray_img, loop_match_img;
		cv::Mat old_img = old_kf->image;
		cv::hconcat(image, gap_image, gap_image);//gap_image = image + gap_image 
		cv::hconcat(gap_image, old_img, gray_img);//gray_img = gap_image + old_img
		cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);//转化颜色空间
		for(int i = 0; i< (int)matched_2d_cur.size(); i++)
		{
			cv::Point2f cur_pt = matched_2d_cur[i];
			cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
		}
		for(int i = 0; i< (int)matched_2d_old.size(); i++)
		{
			cv::Point2f old_pt = matched_2d_old[i];
			old_pt.x += (COL + gap);
			cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
		}
		for (int i = 0; i< (int)matched_2d_cur.size(); i++)
		{
			cv::Point2f old_pt = matched_2d_old[i];
			old_pt.x += (COL + gap) ;
			cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);//画连线
		}
		cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
		putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

		putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
		cv::vconcat(notation, loop_match_img, loop_match_img);

		ostringstream path;
		path <<  "/home/lab606/data/orb_vins/2-25/match/"
				<< index << "-"
				<< old_kf->index << "-" << "before_PnPRANSAC_3pnp_match.jpg";
		cv::imwrite( path.str().c_str(), loop_match_img);
		cout<<"(int)matched_2d_cur.size()deforeRansac:"<<(int)matched_2d_cur.size()<<endl;

	    PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
		cout<<"(int)matched_2d_cur.size()afterRansac:"<<(int)matched_2d_cur.size()<<endl;
	    reduceVector(matched_2d_cur, status);
		cout<<"(int)matched_2d_cur.size()reduceVector:"<<(int)matched_2d_cur.size()<<endl;
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);
	    reduceVector(matched_id, status);
	    #if 1
	    	if (DEBUG_IMAGE)//是否可视化
	        {
				
	        	int gap = 10;
	        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
	            cv::Mat gray_img, loop_match_img;
	            cv::Mat old_img = old_kf->image;
	            cv::hconcat(image, gap_image, gap_image);//gap_image = image + gap_image 
	            cv::hconcat(gap_image, old_img, gray_img);//gray_img = gap_image + old_img
	            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);//转化颜色空间
				//rviz中将matched_2d_cur和matched_2d_old中的点标注出来
				
	            for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	            {
	                cv::Point2f cur_pt = matched_2d_cur[i];
	                cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	            }
	            for(int i = 0; i< (int)matched_2d_old.size(); i++)
	            {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap);
	                cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	            }
	            for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	            {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap) ;
	                cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);//画连线
	            }
	            cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
	            putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

	            putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
	            cv::vconcat(notation, loop_match_img, loop_match_img);

	            
	            ostringstream path;
	            path <<  "/home/lab606/data/orb_vins/2-25/match/"
	                    << index << "-"
	                    << old_kf->index << "-" << "3pnp_match.jpg";
	            cv::imwrite( path.str().c_str(), loop_match_img);
	            if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	            {
	            	/*
	            	cv::imshow("loop connection",loop_match_img);  
	            	cv::waitKey(10);  
	            	*/
	            	cv::Mat thumbimage;
	            	cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
	    	    	sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
	                msg->header.stamp = ros::Time(time_stamp);
	    	    	pub_match_img.publish(msg);
	            }
	        }
	    #endif
		
	}
}

//寻找与建立关键帧和回环帧之间的匹配关系
bool KeyFrame::findConnection(KeyFrame* old_kf)//为了计算相对位姿，最主要的就是利用pnpRansac函数，负责将匹配好的点发布到vins
{
	cout<<"执行findConnection()"<<endl;
	TicToc tmp_t;
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<uchar> status;
	//当前帧的数据
	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	TicToc t_match;

	//历史帧的数据,根据status对无法跟踪的角点进行删除
	searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);//当前关键帧和回环帧进行Brief描述子匹配	
	//根据status剔除匹配失败的点
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	status.clear();//把状态向量清空

	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;

	
    //使用brief特征进行处理
	#if 1
		if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)//25	

		{
			status.clear();
			//添加测试代码
			#if 1 
				//添加测试代码
				cout<<index<<"  :   "<<old_kf->index<<endl;
				//执行RANSAC处理之前
				int gap = 10;
				cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
				cv::Mat gray_img, loop_match_img;
				cv::Mat old_img = old_kf->image;
				cv::hconcat(image, gap_image, gap_image);//gap_image = image + gap_image 
				cv::hconcat(gap_image, old_img, gray_img);//gray_img = gap_image + old_img
				cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);//转化颜色空间
				for(int i = 0; i< (int)matched_2d_cur.size(); i++)
				{
					cv::Point2f cur_pt = matched_2d_cur[i];
					cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
				}
				for(int i = 0; i< (int)matched_2d_old.size(); i++)
				{
					cv::Point2f old_pt = matched_2d_old[i];
					old_pt.x += (COL + gap);
					cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
				}
				for (int i = 0; i< (int)matched_2d_cur.size(); i++)
				{
					cv::Point2f old_pt = matched_2d_old[i];
					old_pt.x += (COL + gap) ;
					cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);//画连线
				}
				cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
				putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

				putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
				cv::vconcat(notation, loop_match_img, loop_match_img);

				ostringstream path;
				path <<  "/home/lab606/data/3-11/d1_map_2/match/"
						<< index << "-"
						<< old_kf->index << "-" << "brief_before_PnPRANSAC_3pnp_match.jpg";
				cv::imwrite( path.str().c_str(), loop_match_img);
				//添加测试代码

			#endif 

			PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);//检测去除误匹配的点
			cout<<"(int)matched_2d_cur.size()afterRansac:"<<(int)matched_2d_cur.size()<<endl;
			reduceVector(matched_2d_cur, status);
			cout<<"(int)matched_2d_cur.size()reduceVector:"<<(int)matched_2d_cur.size()<<endl;
			reduceVector(matched_2d_old, status);
			reduceVector(matched_2d_cur_norm, status);
			reduceVector(matched_2d_old_norm, status);
			reduceVector(matched_3d, status);
			reduceVector(matched_id, status);
			#if 1
				if (DEBUG_IMAGE)//是否可视化
				{
					
					int gap = 10;
					cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
					cv::Mat gray_img, loop_match_img;
					cv::Mat old_img = old_kf->image;
					cv::hconcat(image, gap_image, gap_image);//gap_image = image + gap_image 
					cv::hconcat(gap_image, old_img, gray_img);//gray_img = gap_image + old_img
					cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);//转化颜色空间
					//rviz中将matched_2d_cur和matched_2d_old中的点标注出来
					
					for(int i = 0; i< (int)matched_2d_cur.size(); i++)
					{
						cv::Point2f cur_pt = matched_2d_cur[i];
						cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
					}
					for(int i = 0; i< (int)matched_2d_old.size(); i++)
					{
						cv::Point2f old_pt = matched_2d_old[i];
						old_pt.x += (COL + gap);
						cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
					}
					for (int i = 0; i< (int)matched_2d_cur.size(); i++)
					{
						cv::Point2f old_pt = matched_2d_old[i];
						old_pt.x += (COL + gap) ;
						cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);//画连线
					}
					cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
					putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

					putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
					cv::vconcat(notation, loop_match_img, loop_match_img);

					
					ostringstream path;
					path <<  "/home/lab606/data/3-11/d1_map_2/match/"
							<< index << "-"
							<< old_kf->index << "-" << "brief_pnp_match.jpg";
					cv::imwrite( path.str().c_str(), loop_match_img);
					if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
					{
						/*
						cv::imshow("loop connection",loop_match_img);  
						cv::waitKey(10);  
						*/
						cv::Mat thumbimage;
						cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
						sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
						msg->header.stamp = ros::Time(time_stamp);
						pub_match_img.publish(msg);//绘制回环匹配图像，发布到pub_match_img
					}
				}
			#endif
		}
		if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)//25开始重定位
		{
			
			relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
			relative_q = PnP_R_old.transpose() * origin_vio_R;
			relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());

			if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)//相对偏移角小于30度，且相对位移小于20m，相对位姿检测
			{
				has_loop = true;
				loop_index = old_kf->index;
				//将当前帧和回环帧的相对位姿存入loop_info
				loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
							relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
							relative_yaw;

				//快速重定位，利用该方式修正的位姿，只利用一帧图像来约束
				if(FAST_RELOCALIZATION)
				//msg_match_points 发送给estimator的匹配信息
				//msg_match_points 发送给estimator的匹配信息

				//points里匹配到的角点的归一化坐标和该地点的id
				//channels是回环帧的pose
				{
					sensor_msgs::PointCloud msg_match_points;
					//发送给estimator的匹配信息.ros将sensor_msgs::PointCloud类型的数据传递给vins
					msg_match_points.header.stamp = ros::Time(time_stamp);
					for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
					{
						geometry_msgs::Point32 p;
						p.x = matched_2d_old_norm[i].x;
						p.y = matched_2d_old_norm[i].y;
						p.z = matched_id[i];
						msg_match_points.points.push_back(p);
					}
					Eigen::Vector3d T = old_kf->T_w_i; 
					Eigen::Matrix3d R = old_kf->R_w_i;
					Quaterniond Q(R);
					sensor_msgs::ChannelFloat32 t_q_index;
					t_q_index.values.push_back(T.x());
					t_q_index.values.push_back(T.y());
					t_q_index.values.push_back(T.z());
					t_q_index.values.push_back(Q.w());
					t_q_index.values.push_back(Q.x());
					t_q_index.values.push_back(Q.y());
					t_q_index.values.push_back(Q.z());
					t_q_index.values.push_back(index);
					msg_match_points.channels.push_back(t_q_index);
					//cout<<"findConnection():===================================================发布msg_match_points"<<endl;
					pub_match_points.publish(msg_match_points);
				}
				return true;//如果检测到的当前帧符合要求
			}
		}
	#endif 

	//使用orb特征进行处理
	#if 0
		if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)//25	
		{
			using namespace cv;
			//-- 读取图像
			Mat img_1 = image;
			Mat img_2 = old_kf->image;


			//-- 初始化
			std::vector<KeyPoint> keypoints_1, keypoints_2;                               //定义关键点
			Mat descriptors_1, descriptors_2;                                             //定义描述子
			Ptr<FeatureDetector> detector = ORB::create();                           //Oriented Fast角点位置检测器
			Ptr<DescriptorExtractor> descriptor = ORB::create();                     //描述子检测器
			Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");//匹配器

			//-- 第一步:检测 Oriented FAST 角点位置
			detector->detect(img_1, keypoints_1);                                         //提取图像中的关键点
			detector->detect(img_2, keypoints_2);

			//-- 第二步:根据角点位置计算 BRIEF 描述子
			descriptor->compute(img_1, keypoints_1, descriptors_1);                        //计算图像中的角点的描述子
			descriptor->compute(img_2, keypoints_2, descriptors_2);



			//-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
			vector<DMatch> matches;
			//BFMatcher matcher ( NORM_HAMMING );
			matcher->match(descriptors_1, descriptors_2, matches);                          //匹配两幅图中的描述子

			//-- 第四步:匹配点对筛选
			double min_dist = 10000, max_dist = 0;

			//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
			for (int i = 0; i < descriptors_1.rows; i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}

			min_dist = min_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance; })->distance;
			max_dist = max_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance; })->distance;
			
			//cout<<"-- Max dist : %f "<<max_dist<<endl;
			//cout<<"-- Min dist : %f "<<min_dist<<endl;

			// //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
			// std::vector< DMatch > good_matches;
			// for (int i = 0; i < descriptors_1.rows; i++)
				
			// {
			// 	if (matches[i].distance <= max(1.25 * min_dist,))
			// 	{
			// 		good_matches.push_back(matches[i]);
			// 	}
			// }


			//当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候符合条件的帧数很多，最多25个.
				std::vector< DMatch > good_matches;
				for (int i = 0; i < descriptors_1.rows; i++)
					for (int j = i + 1; j < descriptors_1.rows; j++)
					{
						if (matches[i].distance > matches[j].distance)
						{
							auto tmp = matches[i];
							matches[i] = matches[j];
							matches[j] = tmp;
						}
					}

				for (int i = 0; i < descriptors_1.rows; i++)
				{
					//std::cout << matches[i].distance << std::endl;
					if (matches[i].distance < 2 * min_dist && i < 25)
					{
						good_matches.push_back(matches[i]);
					}
					else
						break;
				}


			//-- 第五步:绘制匹配结果
			Mat img_match;
			Mat img_goodmatch;
			drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
			drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);

			cv::imwrite( "/home/lab606/data/orb_vins/2-27/match/"+to_string(index)+"-"+to_string(old_kf->index)+"-"+"orb_firstmatch.jpg", img_match);
			cv::imwrite( "/home/lab606/data/orb_vins/2-27/match/"+to_string(index)+"-"+to_string(old_kf->index)+"-"+"orb_goodmatch.jpg", img_goodmatch);

			//利用堆积约束求解相机相对位姿变换
			Mat R;
			Mat t;
			// 相机内参,
			Mat K = ( Mat_<double> ( 3,3 ) << 358.47442850029023, 0, 388.40661559633401, 0, 359.52665535350462, 254.76941553631312, 0, 0, 1 );

			//-- 把匹配点转换为vector<Point2f>的形式
			vector<Point2f> points1;
			vector<Point2f> points2;

			for ( int i = 0; i < ( int ) matches.size(); i++ )
			{
				points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
				points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
			}

			//-- 计算基础矩阵
			Mat fundamental_matrix;
			fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
			//cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

			//-- 计算本质矩阵
			Point2d principal_point ( 388.40661559633401,254.76941553631312 );    //相机光心, TUM dataset标定值
			double focal_length = 359;            //相机焦距, TUM dataset标定值
			Mat essential_matrix;
			essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
			//cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

			//-- 计算单应矩阵
			Mat homography_matrix;
			homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
			//cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

			//-- 从本质矩阵中恢复旋转和平移信息.
			recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
			cout<<"R is "<<endl<<R<<endl;
			cout<<"t is "<<endl<<t<<endl;
			cv::cv2eigen(R,PnP_R_old);
			cv::cv2eigen(t,PnP_T_old);

			relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
			relative_q = PnP_R_old.transpose() * origin_vio_R;
			relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
			cout<<abs(relative_yaw)<<endl;
			cout<<relative_t.norm()<<endl;
			//if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)//相对偏移角小于30度，且相对位移小于20m
			{
				has_loop = true;
				loop_index = old_kf->index;
				loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
							relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
							relative_yaw;

				//快速重定位，利用该方式修正的位姿，只利用一帧图像来约束
				if(FAST_RELOCALIZATION)
				//msg_match_points 发送给estimator的匹配信息
				//msg_match_points 发送给estimator的匹配信息

				//points里匹配到的角点的归一化坐标和该地点的id
				//channels是回环帧的pose
				{
					sensor_msgs::PointCloud msg_match_points;
					//发送给estimator的匹配信息.ros将sensor_msgs::PointCloud类型的数据传递给vins
					msg_match_points.header.stamp = ros::Time(time_stamp);
					for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
					{
						geometry_msgs::Point32 p;
						p.x = matched_2d_old_norm[i].x;
						p.y = matched_2d_old_norm[i].y;
						p.z = matched_id[i];
						msg_match_points.points.push_back(p);
					}
					Eigen::Vector3d T = old_kf->T_w_i; 
					Eigen::Matrix3d R = old_kf->R_w_i;
					Quaterniond Q(R);
					sensor_msgs::ChannelFloat32 t_q_index;
					t_q_index.values.push_back(T.x());
					t_q_index.values.push_back(T.y());
					t_q_index.values.push_back(T.z());
					t_q_index.values.push_back(Q.w());
					t_q_index.values.push_back(Q.x());
					t_q_index.values.push_back(Q.y());
					t_q_index.values.push_back(Q.z());
					t_q_index.values.push_back(index);
					msg_match_points.channels.push_back(t_q_index);
					cout<<"通过ORB特征和orb进行重定位====================================================="<<endl;
					pub_match_points.publish(msg_match_points);
				}
				return true;//如果检测到的当前帧符合要求
			}
		}
	#endif
	

	//printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());	
	return false;
}

//计算描述子之间的距离

int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		//printf("update loop info\n");
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}


