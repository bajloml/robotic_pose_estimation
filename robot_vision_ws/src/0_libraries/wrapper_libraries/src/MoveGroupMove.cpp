
#include "wrapper_libraries/MoveGroupMove.h"

MoveGroupMove::MoveGroupMove(const std::string &planningGroup,
                             const std::string &referenceFrame,
                             const std::string &endEffectorLink,
                             const std::string &robotName,
                             const std::string &plannerId):  
    _move_group(_interfaceOptions),
    _interfaceOptions( _planningGroup, _robotName + "/robot_description", _nh), // robot desription under the namespace, under which the actions servers are running
    _nh(_robotName),            
    _planningGroup{planningGroup},
    _referenceFrame{referenceFrame},
    _endEffectorLink{endEffectorLink}, 
    _robotName{robotName},
    _plannerId{plannerId}{

        _move_group.setPlanningTime(10);
        _move_group.setNumPlanningAttempts(2);
        _move_group.allowReplanning(true);
        
        //set EEF link
        /*_move_group.setEndEffectorLink(_endEffectorLink); */
        
        _move_group.setGoalPositionTolerance(0.001);
        _move_group.setGoalOrientationTolerance(0.001);
        _move_group.setMaxVelocityScalingFactor(0.02);
        
        _move_group.setPlannerId(_plannerId);
    }

MoveGroupMove::~MoveGroupMove(){}

const ptree& MoveGroupMove::empty_ptree(){
    static ptree t;
    return t;
}

std::vector<double> MoveGroupMove::getJointStates(){

    return _move_group.getCurrentJointValues();
}

robot_state::RobotStatePtr MoveGroupMove::getCurrentRobotState(){

    return _move_group.getCurrentState();
}

void MoveGroupMove::moveUsingSetPoseTarget(const geometry_msgs::Pose &pose, moveit::planning_interface::MoveItErrorCode &error_code){

    moveit::core::RobotStatePtr robotStatePtr = _move_group.getCurrentState();
    const robot_state::JointModelGroup *joint_model_group = robotStatePtr->getJointModelGroup(_planningGroup);

    _move_group.setPoseTarget(pose);
    //_move_group.setJointValueTarget(pose, _endEffectorLink);

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool successPlan = (_move_group.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if(successPlan){
        error_code = _move_group.move();
    }
    else{
        ROS_WARN("move group, unsuccessful planing");
    }
}

void MoveGroupMove::moveUsingSetPoseTargets(const std::vector<geometry_msgs::Pose> &poseVector, moveit::planning_interface::MoveItErrorCode &error_code){

    moveit::core::RobotStatePtr robotStatePtr = _move_group.getCurrentState();
    const robot_state::JointModelGroup *joint_model_group = robotStatePtr->getJointModelGroup(_planningGroup);

    _move_group.setPoseTargets(poseVector);
    //_move_group.setJointValueTarget(pose, _endEffectorLink);

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool successPlan = (_move_group.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if(successPlan){
        error_code = _move_group.move();
    }
    else{
        ROS_WARN("move group, unsuccessful planing");
    }
}

void MoveGroupMove::moveUsingCartesianPath(const std::vector< geometry_msgs::Pose > &waypoints,
                                           moveit_msgs::RobotTrajectory &trajectory_msg, 
                                           double eef_step, 
                                           double jump_threshold){

    moveit::core::RobotStatePtr robotStatePtr = _move_group.getCurrentState();
    const robot_state::JointModelGroup *joint_model_group = robotStatePtr->getJointModelGroup(_planningGroup);
    double fraction = _move_group.computeCartesianPath( waypoints,
                                                        eef_step,
                                                        jump_threshold,
                                                        trajectory_msg, 
                                                        false);
    ROS_WARN("Cartesian trajectory was succesfully planned %.2f%%", fraction * 100.0);                                                    
    if(fraction==1.0){
        ROS_WARN("Cartesian planning was succesfull. Moving...");
        _move_group.execute(trajectory_msg);
    }
    else{
        ROS_WARN("Cartesian planning was unsuccesful!");
    }
}

//approach to target using moveit grasp
void MoveGroupMove::moveUsingGrasp( std::string ROBOT_NAME, 
                                    std::string REFERENCE_FRAME, 
                                    const geometry_msgs::Pose &wantedPose, 
                                    const char* approachAxis,
                                    double approachDirection,
                                    const char* retreatAxis,
                                    double retreatDirection)
{
    std::vector<moveit_msgs::Grasp> grasps;
    grasps.resize(1);

    grasps[0].grasp_pose.header.frame_id = ROBOT_NAME + "/" + REFERENCE_FRAME;
    grasps[0].grasp_pose.pose.orientation = wantedPose.orientation;
    grasps[0].grasp_pose.pose.position = wantedPose.position;

    /* Set approach parameters */
    /* Defined with respect to frame_id */
    grasps[0].pre_grasp_approach.direction.header.frame_id = ROBOT_NAME + "/" + REFERENCE_FRAME;
    
    if (approachDirection!=1.0 && approachDirection !=-1.0)
        return;

    switch (*approachAxis)
    {
        case 'x':
        case 'X':
            grasps[0].pre_grasp_approach.direction.vector.x = approachDirection;
            break;
        case 'y':
        case 'Y':
            grasps[0].pre_grasp_approach.direction.vector.y = approachDirection;
            break;
        case 'z':
        case 'Z':
            grasps[0].pre_grasp_approach.direction.vector.z = approachDirection;
            break;
        default:
            std::cout << "approachAxis must be set to x,y or z " << std::endl;
            return;
    }

    grasps[0].pre_grasp_approach.min_distance = 0.1;      //0.095
    grasps[0].pre_grasp_approach.desired_distance = 0.15;   //0.215

    /* Set reterat parameters */
    /* Defined with respect to frame_id */
    grasps[0].post_grasp_retreat.direction.header.frame_id =  ROBOT_NAME + "/" + REFERENCE_FRAME;

    if (retreatDirection!=1.0 && retreatDirection !=-1.0)
        return;

    switch (*retreatAxis)
    {
        case 'x':
        case 'X':
            grasps[0].post_grasp_retreat.direction.vector.x = retreatDirection;
            break;
        case 'y':
        case 'Y':
            grasps[0].post_grasp_retreat.direction.vector.y = retreatDirection;
            break;
        case 'z':
        case 'Z':
            grasps[0].post_grasp_retreat.direction.vector.z = retreatDirection;
            break;
        default:
            std::cout << "retreatAxis must be set to x,y or z " << std::endl;
            return;
    }
    // Setting post-grasp retreat
    // ++++++++++++++++++++++++++

    grasps[0].post_grasp_retreat.min_distance = 0.1;
    grasps[0].post_grasp_retreat.desired_distance = 0.15;

    // Set support surface as table1.
    _move_group.setSupportSurfaceName("table1");
    // Call pick to pick up the object using the grasps given
    _move_group.pick("object", grasps);
}

//read "home" position from srdf file and send robot there
void MoveGroupMove::toHome(std::string srdf_path){

    std::vector<double> joints;
    ptree pt;

    try{
        read_xml(srdf_path, pt, boost::property_tree::xml_parser::trim_whitespace);

        const ptree & joints_xml = pt.get_child("robot.group_state", empty_ptree());

        BOOST_FOREACH (const ptree::value_type & node, joints_xml){

            const ptree & attributes = node.second.get_child("<xmlattr>", empty_ptree());

            BOOST_FOREACH(const ptree::value_type &v, attributes){
                if(std::string(v.first.data()) == "value"){
                    joints.push_back(std::stod(v.second.data()));
                }
            }
        }

        //send move group to home position
        _move_group.setJointValueTarget(joints);
        _move_group.move();

    }
    catch (...){
        std::cout << "Given SRDF file doesn't exist!!!"<<std::endl;
    }
}

void MoveGroupMove::moveUsingJointState(geometry_msgs::Pose &pose){

    robot_model_loader::RobotModelLoader robot_model_loader(_robotName + "/robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();

    //Create a RobotState and JointModelGroup to keep track of the current robot pose and planning group
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    const robot_state::JointModelGroup* joint_model_group = robot_state->getJointModelGroup(_planningGroup);

    const std::vector<std::string>& joint_names = joint_model_group->getVariableNames();
    std::vector<double> joint_values;

    ROS_INFO("checking inverse kinematics");
    if(robot_state->setFromIK(joint_model_group, pose)){
        ROS_INFO("IK success!");
        robot_state->copyJointGroupPositions(joint_model_group, joint_values);
        _move_group.setJointValueTarget(joint_values);
    }
    else{
        ROS_ERROR("IK FAILED");
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (_move_group.plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    _move_group.execute(plan);

    return;

}


