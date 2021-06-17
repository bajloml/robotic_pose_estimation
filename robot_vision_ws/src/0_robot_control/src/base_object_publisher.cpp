// C++
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <math.h>
//bridge between ROS and opencv; used for reading from camera topic in ROS and giving it to opencv
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2/transform_datatypes.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/buffer_core.h>
//yaml parser
#include <yaml-cpp/yaml.h>

std::string getCurrentPath(){
  char buff[PATH_MAX];
  getcwd( buff, PATH_MAX );
  std::string cwd( buff );
  return cwd;
}

Eigen::Quaterniond euler2Quaternion( const double roll,
                                     const double pitch,
                                     const double yaw )
{
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    return q;
}

void create_gmTransform(float x, float y, float z, 
                        float roll, float pitch, float yaw, 
                        tf2::Stamped<tf2::Transform> &_tf2_trans)
{
  //create Eigen translation vector, Eigen 4x4 affine matrix and fill matrix with vector and euler angles
  Eigen::Vector3d trans(x,y,z);
  Eigen::Affine3d _eMc= Eigen::Affine3d::Identity();
  _eMc.translation() = trans;
  _eMc.linear() = (euler2Quaternion(roll, pitch, yaw)).toRotationMatrix();

  geometry_msgs::TransformStamped _gm_Trans = tf2::eigenToTransform(_eMc);

  tf2::fromMsg(_gm_Trans, _tf2_trans);
}


void poseToTransform(const geometry_msgs::Pose &pose, geometry_msgs::TransformStamped &transform, std::string childFrame, std::string parentFrame)
{
  transform.transform.rotation.x = pose.orientation.x;
  transform.transform.rotation.y = pose.orientation.y;
  transform.transform.rotation.z = pose.orientation.z;
  transform.transform.rotation.w = pose.orientation.w;

  transform.transform.translation.x = static_cast<double>(pose.position.x);
  transform.transform.translation.y = static_cast<double>(pose.position.y);
  transform.transform.translation.z = static_cast<double>(pose.position.z);

  transform.child_frame_id = childFrame;
  transform.header.frame_id = parentFrame;
  transform.header.stamp = ros::Time::now();
}



/**  Main program  **/
int main(int argc, char *argv[])
{
    try
    {
        std::cout << "Current directory is: " << getCurrentPath() << std::endl;

        /*ONLY FOR DEBUG  */
        //std::string PLANNING_GROUP     = "robot_arm";
        std::string _END_EFFECTOR_LINK  = "tool_gripper_tip";
        std::string _CAMERA_LINK        = "fixed_camera_link";
        std::string _CAMERA_FRAME       = "camera";
        std::string _OBJECT             = "thor";
        std::string _OBJECT_TO_PUBLISH  = "thor_in_base";
        std::string _REFERENCE_FRAME    = "base_link";
        std::string _ROBOT_NAME         = "fanuc_1";

        ros::init(argc, argv, "base_object_publisher");
        //ros::NodeHandle n;
        ros::NodeHandlePtr n = boost::make_shared<ros::NodeHandle>();
        ros::Rate loop_rate(50);
        ros::AsyncSpinner spinner(5); 
        spinner.start();

        /*  get the parameters given to the node and assign them to the internal variables   */
        std::string _nodeName = ros::this_node::getName();
        _nodeName.erase(0,1);
        n->getParam(_nodeName + "/END_EFFECTOR_LINK", _END_EFFECTOR_LINK);
        n->getParam(_nodeName + "/CAMERA_LINK",       _CAMERA_LINK);
        n->getParam(_nodeName + "/CAMERA_FRAME",      _CAMERA_FRAME);
        n->getParam(_nodeName + "/OBJECT",            _OBJECT);
        n->getParam(_nodeName + "/OBJECT_TO_PUBLISH", _OBJECT_TO_PUBLISH);
        n->getParam(_nodeName + "/REFERENCE_FRAME",   _REFERENCE_FRAME);
        n->getParam(_nodeName + "/ROBOT_NAME",        _ROBOT_NAME);

        tf2_ros::Buffer                 _tfBuffer;
        tf2_ros::TransformListener      _tfListener(_tfBuffer);
        geometry_msgs::TransformStamped _ts_base_camera, _ts_camera_object;
        geometry_msgs::Pose             _wantedPose, _pre_wantedPose, _homePose;
        //tf2::Stamped<tf2::Transform>    _tf2_eMc;

        //broadcast transformation so it could be visualized in RViz
        tf2_ros::TransformBroadcaster   _br;
        geometry_msgs::TransformStamped _objectTransform;

        //camera adaption transform needed because camera link is rotated and that 
        //rotation has to be be included in calculation for object position
        tf2::Stamped<tf2::Transform> _tf2_adaptionFixedCamera1;
        create_gmTransform(0, 0, 0, 0, 0, 1.57, _tf2_adaptionFixedCamera1);

        Eigen::Isometry3d _text_pose = Eigen::Isometry3d::Identity();
  
        while (ros::ok() && n->ok())
        { 
            while(    _tfBuffer.canTransform(_REFERENCE_FRAME, _CAMERA_LINK, ros::Time(0), ros::Duration(1))
                  &&  _tfBuffer.canTransform(_CAMERA_FRAME, _OBJECT, ros::Time(0), ros::Duration(1))
                  &&  _tfBuffer.canTransform(_REFERENCE_FRAME, _END_EFFECTOR_LINK, ros::Time(0), ros::Duration(1))
                  )
            {
                //get transform between robot base_link and camera link
                _ts_base_camera = _tfBuffer.lookupTransform(_REFERENCE_FRAME, _CAMERA_LINK, ros::Time(0));

                //calculate transform between base and camera
                tf2::Stamped<tf2::Transform> _tf2_ts_base_camera;
                tf2::fromMsg(_ts_base_camera, _tf2_ts_base_camera);

                //get transform between camera_link and object
                _ts_camera_object = _tfBuffer.lookupTransform( _CAMERA_FRAME, _OBJECT, ros::Time(0));

                //calculate transform between camera and object
                tf2::Stamped<tf2::Transform> _tf2_ts_camera_object;
                tf2::fromMsg(_ts_camera_object, _tf2_ts_camera_object);

                tf2::Transform _tf2_ts_base_object = _tf2_ts_base_camera * _tf2_adaptionFixedCamera1 * _tf2_ts_camera_object;
                
                tf2::toMsg(_tf2_ts_base_object, _wantedPose);

                //publish _wantedPose so it can be visualized in RViz as marker
                poseToTransform( _wantedPose, _objectTransform, _OBJECT_TO_PUBLISH, _REFERENCE_FRAME);

                //send transform
                _br.sendTransform(_objectTransform);

                loop_rate.sleep();
            }
        }
      }

    catch(tf2::TransformException &ex){
        std::cout<<" tf2::TransformException:"<< std::string(ex.what())<<std::endl;
    }
    catch(std::exception e){
        std::cout<<" int main:"<< std::string(e.what())<<std::endl;
    }
  return 0;
}