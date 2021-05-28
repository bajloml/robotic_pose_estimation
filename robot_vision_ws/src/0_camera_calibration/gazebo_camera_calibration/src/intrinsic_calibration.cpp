#include <wrapper_libraries/Intrinsic_Cal.h>

int main(int argc, char** argv){

    std::string pathToImages        = "";       //  path to the dir where the images are
    std::string savePath            = "";       //  path to the dir where the camera_settings.json should be saved
    std::string node_name           = "";       //  name of the node
    bool showImages                 = false;    //  should the image be shown during the calibration

    char ch;
    char buff[100];
    unsigned cpt = 0;

    //ros initialization
    ros::init(argc, argv, "intrinsic_calibration_chessboard");
    ros::NodeHandle node;
    ros::Rate rate(10.0);

    node_name = ros::this_node::getName();
    ROS_INFO("\nintrinsic_calibration_node name is: %s",node_name.c_str());

    //get current working directory
    getcwd(buff, 100);
    std::string currentWorkingDirectory(buff);
    ROS_INFO("\nworking directory path is: %s",currentWorkingDirectory.c_str());

    /* get the parameters given to the node and assign them to the internal variables  */
    node.getParam(node_name + "/pathToImages", pathToImages);
    node.getParam(node_name + "/savePath", savePath);
    node.getParam(node_name + "/showImages", showImages);

    if (pathToImages.compare("")!=0){ROS_INFO("fromFrame: %s", pathToImages.c_str());}
    else{ROS_ERROR("Failed to get param <pathToImages>");return 0;exit(0);}
    if (savePath.compare("")!=0){ROS_INFO("\nyaml files path is: %s",savePath.c_str());}
    else{ROS_ERROR("\nFailed to get the <savePath>");return 0;}
    
    /*  USED JUST FOR DEBUGGING     
    pathToImages = "/home/ros-industrial/Desktop/ROS_WS/src/0_camera_calibration/output_extrinsic_calibration_images_poses/*.png";
    savePath = "/home/ros-industrial/Desktop/ROS_WS/src/0_camera_calibration/output_intrinsic_calibration/";
    */
   
    // declare an object for a calibration
    Intrinsic_Cal intrinsic_calibration(showImages);

    while (node.ok() && ros::ok()){
        
        intrinsic_calibration.calibrate(pathToImages);
        intrinsic_calibration.saveSettings(savePath);
        ROS_INFO("DONE");

        return 0;
    }
}