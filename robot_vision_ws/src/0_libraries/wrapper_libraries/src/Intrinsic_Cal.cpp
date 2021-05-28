#include <wrapper_libraries/Intrinsic_Cal.h>

Intrinsic_Cal::Intrinsic_Cal(bool showImages):showImages_(showImages){

    // get corner coords
    for(int i{0}; i<chessboard_dim_[1]; i++){
        for(int j{0}; j<chessboard_dim_[0]; j++)
        coords_3D_.push_back(cv::Point3f(j,i,0));
    }

}

Intrinsic_Cal::~Intrinsic_Cal(){
}

void Intrinsic_Cal::calibrate(std::string pathToImages){

    ROS_INFO("reading images from directory: %s", pathToImages.c_str());
    cv::glob(pathToImages, imagesPath_, false);

    // Looping over all the images in the directory
    for(int i=0; i<imagesPath_.size(); i++){

        ROS_INFO("Reading: %s", imagesPath_[i].c_str());

        img_ = cv::imread(imagesPath_[i]);
        cv::cvtColor(img_, img_gray_, cv::COLOR_BGR2GRAY);

        // find corners, success --> correct number of corners found
        succ_ = cv::findChessboardCorners(img_gray_, cv::Size(chessboard_dim_[0], chessboard_dim_[1]), coords_2D_, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
        
        if(succ_){
            cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 1e-10);
            
            // refining pixel coordinates for given 2d points.
            cv::cornerSubPix(img_gray_, coords_2D_, cv::Size(11,11), cv::Size(-1,-1), criteria);
            
            // Displaying the detected corner points on the checker board
            cv::drawChessboardCorners(img_, cv::Size(chessboard_dim_[0], chessboard_dim_[1]), coords_2D_, succ_);
            
            points_3D_.push_back(coords_3D_);
            points_2D_.push_back(coords_2D_);
        }

        // show images for debug
        if (showImages_){
            cv::imshow(imagesPath_[i], img_);
            cv::waitKey(0);
        }
    }

    cv::destroyAllWindows();

    // calibrate with 3D points and assigned 2D points
    // it changes variables assigned to outputs, intrinsic parameters and extrinsic parameters(Rot and Trans) for each image(view)
    // returns RMS error(deviation)
    RMS_ = cv::calibrateCamera(points_3D_, points_2D_, cv::Size(img_gray_.rows, img_gray_.cols), camMatrix_, dist_, R_, T_);
    ROS_INFO("Calibration done, RMS = %f", RMS_);

    return;
}

void Intrinsic_Cal::saveSettings(std::string savePath){
    // save camera settings on the given path
    cv::FileStorage cam_settigs_storage(savePath+"cam_settings"+".json", cv::FileStorage::Mode::WRITE);
    cam_settigs_storage.write("intrinsic_settings", camMatrix_);
    cam_settigs_storage.write("dist", dist_);
    cam_settigs_storage.release();
    ROS_INFO("camera settings saved in: %s", savePath.c_str());
}