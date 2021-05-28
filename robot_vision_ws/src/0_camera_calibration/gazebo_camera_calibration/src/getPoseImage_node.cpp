//  ros includes
#include <ros/ros.h>

#include"wrapper_libraries/imageSaver.h"
#include"wrapper_libraries/PoseSaver.h"
#include"wrapper_libraries/MarkerDebug.h"


int main(int argc, char** argv){
    
    std::string fromFrame           = "";       //  transform from which frame
    std::string toFrame             = "";       //  transform to which frame
    std::string savePath            = "";       //  where to save yaml file path
    std::string cameraImageTopic    = "";       //  camera image topic
    std::string node_name           = "";       //  name of the node
    bool publishPoseMarker          = false;    //  should the node publish the marker in rviz

    char ch;
    char buff[100];
    unsigned cpt = 0;

    //ros initialization
    ros::init(argc, argv, "pose_image_pairs");
    ros::NodeHandle node;
    ros::Rate rate(10.0);

    node_name = ros::this_node::getName();
    ROS_INFO("\ngetPoseImage_node name is: %s",node_name.c_str());

    //get current working directory
    getcwd(buff, 100);
    std::string currentWorkingDirectory(buff);
    ROS_INFO("\nworking directory path is: %s",currentWorkingDirectory.c_str());
  
    /*  get the parameters given to the node and assign them to the internal variables  */
    node.getParam(node_name + "/fromFrame",        fromFrame);
    node.getParam(node_name + "/toFrame",          toFrame);
    node.getParam(node_name + "/savePath",         savePath);
    node.getParam(node_name + "/cameraImageTopic", cameraImageTopic);
    node.getParam(node_name + "/publishPoseMarker", publishPoseMarker);

    if (fromFrame.compare("")!=0){ROS_INFO("fromFrame: %s", fromFrame.c_str());}
    else{ROS_ERROR("Failed to get param <fromFrame>");return 0;exit(0);}
    if (toFrame.compare("")!=0){ROS_INFO("toFrame: %s", toFrame.c_str());}
    else{ROS_ERROR("Failed to get param <toFrame>");return 0;exit(0);}
    if (savePath.compare("")!=0){ROS_INFO("\nyaml files path is: %s",savePath.c_str());}
    else{ROS_ERROR("\nFailed to get the <savePath>");return 0;}
    if (cameraImageTopic.compare("")!=0){ROS_INFO("\ncamera image topic is: %s",cameraImageTopic.c_str());}
    else{ROS_ERROR("\nFailed to get the <cameraImageTopic>");return 0;}     
    
    /*  USED JUST FOR DEBUGGING     
    fromFrame           = "tool0";
    toFrame             = "base_link";
    savePath            = "/home/ros-industrial/Desktop/ROS_WS/src/0_camera_calibration/output_extrinsic_calibration_images_poses/";
    cameraImageTopic    = "/fanuc_1/cameraEyeInHand/image_raw";
    publishPoseMarker   = true;
    */

    // create object to save image, pose and to use markers in rviz to debug
    PoseSaver   poseSaver(savePath);
    ImageSaver  imageSaver(cameraImageTopic, savePath);
    MarkerDebug markerDebug(toFrame);

    // eigne transform, used to show the marker in rviz
    Eigen::Transform<float, 3, Eigen::Affine> transform; 

    while (node.ok() && ros::ok()){
        ROS_INFO("POSE/IMAGE PAIR COUNTER INITIALIZED: %i ", cpt);
        ROS_INFO("TYPE 'n' TO SAVE A POSE/IMAGE PAIR OR 'c' TO CLOSE");
        //wait for the input charachter
        std::cin>>ch;
        if(ch=='n'){
            cpt ++;
            try{
                ros::spinOnce();    // spinOnce() process callbacks only when the 'n' is pressed
                                    // Otherwise if spin() was used, it wouldn't run the rest of the program
                
                // if rviz markers are used to debug
                if (publishPoseMarker){
                    markerDebug.deleteMarkers();
                }

                // get the Eigen transform and save the pose in the yaml file
                transform = poseSaver.saveYamlPoseFile(toFrame, fromFrame, cpt);

                // if rviz markers are used to debug
                if (publishPoseMarker){
                    markerDebug.addMarker(markerDebug.EigenTransToPose(transform), "poseFromEigen");
                }

                // save the camera image in this pose
                imageSaver.saveImage(cpt);
            }
            catch (tf::TransformException &ex) {
                ROS_ERROR("%s",ex.what());
                return 0;
                exit(0);
            }
        }
        else if(ch=='c'){
            return 0;
            exit(0);
        }
    }
}        
