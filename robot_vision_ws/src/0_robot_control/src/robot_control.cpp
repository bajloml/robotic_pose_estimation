// C++
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <boost/filesystem.hpp>
#include <sstream>
#include <string>
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
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
//yaml parser
#include <yaml-cpp/yaml.h>

//moveit
#include "wrapper_libraries/MoveGroupMove.h"
#include "wrapper_libraries/MarkerDebug.h"
//#include "MoveGroupMove_old.h"
#include <geometric_shapes/shape_operations.h>

std::string getCurrentPath()
{
  char buff[PATH_MAX];
  getcwd( buff, PATH_MAX );
  std::string cwd( buff );
  return cwd;
}


/**  Main program  **/
int main(int argc, char *argv[])
{
  try{
    std::cout << "Current directory is: " << getCurrentPath() << std::endl;
    
    std::string _nodeName, _retreatAxis;

    /*ONLY FOR DEBUG  */
    std::string _PLANNING_GROUP     = "robot_arm";
    std::string _APPROACH_AXIS      = "x";
    std::string _SRDF = "/home/ros/Desktop/robot_vision_ws/src/0_robot_and_gripper_models/moveit_robot_gripper_camera/config/fanuc_m16ib20.srdf";
    std::string _PLANNER_ID         = "RRT";
    std::string _END_EFFECTOR_LINK  = "tool_gripper_tip";
    std::string _OBJECT             = "thor";
    std::string _REFERENCE_FRAME    = "base_link";
    std::string _ROBOT_NAME         = "fanuc_1";
    
    ros::init(argc, argv, "robot_control");
    //ros::NodeHandle n;
    ros::NodeHandlePtr n = boost::make_shared<ros::NodeHandle>();
    ros::Rate loop_rate(5);
    ros::AsyncSpinner spinner(3); 
    spinner.start();

    tf2_ros::Buffer                 _tfBuffer;
    tf2_ros::TransformListener      _tfListener(_tfBuffer);
    geometry_msgs::TransformStamped _ts_base_object, _ts_base_eef;

    //offset values
    double _x_offset=0.0, _y_offset=0.0, _z_offset=0.0;

    /*  get the parameters given to the node and assign them to the internal variables   */
    _nodeName = ros::this_node::getName();
    _nodeName.erase(0,1);
    n->getParam(_nodeName + "/PLANNING_GROUP",     _PLANNING_GROUP);
    n->getParam(_nodeName + "/PLANNER_ID",         _PLANNER_ID);
    n->getParam(_nodeName + "/END_EFFECTOR_LINK",  _END_EFFECTOR_LINK);        
    n->getParam(_nodeName + "/REFERENCE_FRAME",    _REFERENCE_FRAME);   
    n->getParam(_nodeName + "/approachAxis",       _APPROACH_AXIS);
    n->getParam(_nodeName + "/srdf_file",          _SRDF);
    n->getParam(_nodeName + "/OBJECT",             _OBJECT);
    n->getParam(_nodeName + "/_x_offset",          _x_offset);
    n->getParam(_nodeName + "/_y_offset",          _y_offset);
    n->getParam(_nodeName + "/_z_offset",          _z_offset);

    MoveGroupMove moveGroupTest(_PLANNING_GROUP, _END_EFFECTOR_LINK, _REFERENCE_FRAME, _ROBOT_NAME, _PLANNER_ID);

    while (ros::ok() && n->ok())
    {
      ros::Duration(3).sleep();
      
      //first send robot to home position
      std::cout << "--------------------------------Moving to home position!--------------------------------------" << std::endl;
      moveGroupTest.toHome( _SRDF);
      std::cout << "----------------------------------------------------------------------------------------------" << std::endl;

      while(ros::ok() && n->ok()){
        
        std::cout << "Enter object name to make move group " << _PLANNING_GROUP << "' attempt" << std::endl;
        std::string input = "";
        std::getline(std::cin, input);
        std::cout << "Trying to move to: " << input << std::endl;

        if(input==_OBJECT){
          std::cout << "Robot movement attempt: " << std::endl;
        }
        else if(input=="home"){
          moveGroupTest.toHome( _SRDF);
          continue;
        }
        else{
          continue;
        }
        //get object in reference frame transform
        if(_tfBuffer.canTransform(_REFERENCE_FRAME, _OBJECT+"_in_base", ros::Time(0), ros::Duration(1.0))){
          _ts_base_object = _tfBuffer.lookupTransform(_REFERENCE_FRAME, _OBJECT+"_in_base", ros::Time(0), ros::Duration(1.0));
        }
        else{
          std::cout << "Transform between " << _REFERENCE_FRAME << " AND " << _OBJECT << "_in_base is unavailable" << std::endl;
          continue;
        }
        //get EEF in reference frame transform
        if(_tfBuffer.canTransform(_REFERENCE_FRAME, _END_EFFECTOR_LINK, ros::Time(0), ros::Duration(1.0))){
          _ts_base_eef = _tfBuffer.lookupTransform(_REFERENCE_FRAME, _END_EFFECTOR_LINK, ros::Time(0), ros::Duration(1.0));
        }
        else{
          std::cout << "Transform between " << _REFERENCE_FRAME << " AND " << _END_EFFECTOR_LINK << "_in_base is unavailable" << std::endl;
          continue;
        }

        //calculate transform between base and object
        tf2::Stamped<tf2::Transform> _tf2_ts_base_object;
        tf2::fromMsg(_ts_base_object, _tf2_ts_base_object);
     
        //move to preposition
        tf2::Stamped<tf2::Transform> preposition = _tf2_ts_base_object;
        tf2::Stamped<tf2::Transform> position = _tf2_ts_base_object;
      
        geometry_msgs::Pose gm_preposition;
        geometry_msgs::Pose gm_position;

        //translate transforms to messages
        tf2::toMsg(preposition, gm_preposition);
        tf2::toMsg(position, gm_position);

        //position offset, just for testing
        gm_preposition.position.x = gm_position.position.x + _x_offset;
        gm_preposition.position.y = gm_position.position.y + _y_offset;
        gm_preposition.position.z = gm_position.position.z + _z_offset;

        tf2::Quaternion q_orig, q_rot1, q_rot2, q_new;

        // Get the original orientation of 'commanded_pose'
        tf2::convert(gm_position.orientation, q_orig);
        // Rotate the previous pose by 90* around X
        double r1=0, p1=0, y1=0;
        double r2=0, p2=0, y2=0;
        if (_APPROACH_AXIS=="x"){
          r1 = 0;
          p1 = M_PI_2;
          y1 = 0;

          r2 = 0;
          p2 = 0;
          y2 = -M_PI_2;
        }
        //not tested
        else if(_APPROACH_AXIS=="y"){
          r1 = M_PI_2;
          p1 = 0;
          y1 = 0;
        }
        //not tested
        else if(_APPROACH_AXIS=="z"){
          r1 = 0;
          p1 = 0;
          y1 = 0;
        }
        
        q_rot1.setRPY(r1, p1, y1);
        q_rot2.setRPY(r2, p2, y2);
        // Calculate the new orientation
        q_new = q_orig*q_rot1*q_rot2; 
        q_new.normalize();

        // Stuff the new rotation back into the pose. This requires conversion into a msg type
        tf2::convert(q_new, gm_position.orientation);
        tf2::convert(q_new, gm_preposition.orientation);
        
        //create vector of the geometry_msgs::PoseStamped messages
        std::vector< geometry_msgs::Pose > wp;
        //fill the vector
        wp.push_back(gm_preposition);
        wp.push_back(gm_position);

        /**************************visualize the poses in the rviz using markers*****************************************/        
        MarkerDebug _marker_debug(_REFERENCE_FRAME);   // has to be in the frame in accordance to which you want to show your points(markers)
        //first delete all markers
        _marker_debug.deleteMarkers();
        //add markers on the plain
        for (int i=0; i<wp.size(); i++){
          _marker_debug.addMarker(wp[i], "pose"+std::to_string(i));
        }

        //using cartesian path
        moveit_msgs::RobotTrajectory trajectory_msg;
        moveGroupTest.moveUsingCartesianPath(wp, trajectory_msg);
        // moveGroupTest.moveUsingJointState(wp[0]);
        // moveGroupTest.moveUsingJointState(wp[1]);

        loop_rate.sleep();
      }
    }
    return 0;
  }
  catch(std::exception e){
    std::cout<<" int main:"<<e.what()<<std::endl;
  }
}