#ifndef Robust_Grasp_H
#define Robust_Grasp_H

#include <ros/ros.h>
#include <kinova_driver/kinova_ros_types.h>

#include <actionlib/client/simple_action_client.h>
#include <kinova_msgs/SetFingersPositionAction.h>

#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <mutex>
#include <control_msgs/JointTrajectoryControllerState.h>

// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>

#include <moveit/move_group_interface/move_group.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/planning_pipeline/planning_pipeline.h>

#include <moveit/kinematic_constraints/utils.h>
#include <geometric_shapes/solid_primitive_dims.h>

#include <moveit_msgs/PlanningScene.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/GetStateValidity.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/ApplyPlanningScene.h>
#include <moveit_msgs/DisplayTrajectory.h>


namespace kinova
{


  class RobustGrasp
  {
  public:
    RobustGrasp(ros::NodeHandle &nh);
    ~RobustGrasp();



  private:
    ros::NodeHandle nh_;

    // open&close fingers: gripper_group_.plan not alway have a solution
    actionlib::SimpleActionClient<kinova_msgs::SetFingersPositionAction>* finger_client_;

    moveit::planning_interface::MoveGroup* group_;
    moveit::planning_interface::MoveGroup* gripper_group_;
    robot_model::RobotModelPtr robot_model_;
    //        robot_state::RobotStatePtr robot_state_;

    planning_scene::PlanningScenePtr planning_scene_;
    planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;

    // work scene
    moveit_msgs::CollisionObject co_;
    moveit_msgs::AttachedCollisionObject aco_;
    moveit_msgs::PlanningScene planning_scene_msg_;


    ros::Publisher pub_co_;
    ros::Publisher pub_aco_;
    ros::Publisher pub_planning_scene_diff_;
    ros::Subscriber sub_pose_;
    ros::Subscriber sub_joint_;

    //
    std::vector<std::string> joint_names_;
    std::vector<double> joint_values_;

    // use Kinova Inverse Kinematic model to generate joint value, then setJointTarget().
    bool use_KinovaInK_;

    // check some process if success.
    bool result_;
    // wait for user input to continue: cin >> pause_;
    std::string pause_;
    std::string robot_type_;

    // update current state and pose
    boost::mutex mutex_state_;
    boost::mutex mutex_pose_;
    sensor_msgs::JointState current_state_;
    geometry_msgs::PoseStamped current_pose_;


    // define pick_place joint value and pose
    std::vector<double> start_joint_;
    std::vector<double> grasp_joint_;
    std::vector<double> pregrasp_joint_;
    std::vector<double> postgrasp_joint_;

    geometry_msgs::PoseStamped start_pose_;
    geometry_msgs::PoseStamped grasp_pose_;
    geometry_msgs::PoseStamped can_pose_;
    geometry_msgs::PoseStamped pregrasp_pose_;
    geometry_msgs::PoseStamped postgrasp_pose_;


    void build_workscene();
    void add_obstacle();
    void add_complex_obstacle();
    void clear_obstacle();
    void clear_workscene();
    void add_attached_obstacle();
    void add_target();

    void define_joint_values();
    void define_cartesian_pose();
    geometry_msgs::PoseStamped generate_gripper_align_pose(geometry_msgs::PoseStamped targetpose_msg, double dist, double azimuth, double polar, double rot_gripper_z);
    void setup_constrain(geometry_msgs::Pose target_pose, bool orientation, bool position);
    void check_constrain();

    bool my_pick();
    bool my_place();

    void get_current_state(const sensor_msgs::JointStateConstPtr &msg);
    void get_current_pose(const geometry_msgs::PoseStampedConstPtr &msg);
    // TODO: use Kinova inverse kinematic solution instead of from ROS.
    void getInvK(geometry_msgs::Pose &eef_pose, std::vector<double> &joint_value);
    void check_collision();
    bool execute_plan(moveit::planning_interface::MoveGroup &group);
    bool gripper_action(double gripper_rad);

    int max_attempts;

    // fixed pitch and yaw
    geometry_msgs::PoseStamped generate_pose(double x, double y, double z, double roll, double pitch, double yaw);

    int run_script();

  public:
    bool robot_connected_;
    double camera_pose_x;
    double camera_pose_y;
    double camera_pose_z;
    bool move_to_homepose();
    moveit::planning_interface::MoveItErrorCode move_arm(geometry_msgs::PoseStamped &poseStamped, int attempts = 1);
    moveit::planning_interface::MoveItErrorCode move_gripper(std::string target, double finger_turn = 0);
    // high level method, go to (x,y,z), grab and comeback
    int grasp_at(double x, double y, double z, bool if_comeback = true, double roll = 0);
    int grasp_at(geometry_msgs::PoseStamped& gasp_pose, bool if_comeback = true, int wait = 1);

    inline geometry_msgs::PoseStamped getCurrentPose() {
      return group_->getCurrentPose();
    }
    inline void stop() {
      return group_->stop();
    }

    bool grasp_randomly(int num);
    void listen_to_console();
    bool start_service();

    /*
      void depthImageCallback(const sensor_msgs::ImageConstPtr& msg);
      void depthCameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& ci);
      void colorImageCallback(const sensor_msgs::ImageConstPtr& msg);
    */
    void imageDisplayCallback(const sensor_msgs::ImageConstPtr& msg);
    ros::ServiceClient planner_service;
    class Image_Buffer {
    public:
      sensor_msgs::Image depth_msg;
      sensor_msgs::Image color_msg;
      sensor_msgs::CameraInfo camera_info;
      long id;
      long last_id;
      bool get_new(cv::Mat &depth, cv::Mat &color);
      bool get_new_msg(sensor_msgs::Image &depth, sensor_msgs::Image &color, sensor_msgs::CameraInfo &info, double & distance, bool convert_from_mm = false);
      bool write_color_and_depth_to(std::string filename);
      std::mutex mutlock;
      Image_Buffer() {
	id = last_id = 0;
      }
    } image_buffer;
    //static void rgbdCallback(const sensor_msgs::CameraInfoConstPtr& ci, const sensor_msgs::ImageConstPtr& de, const sensor_msgs::ImageConstPtr& co, _image_buffer& ib);

    class Joint_Feedback {
    public:
      ros::Time start;
      ros::Time end;
      std::vector<double> errors;
      Joint_Feedback() {
	start = ros::Time::now();
      }
      void reset() {
        errors.resize(0);
      }
      void handle_feedback(const control_msgs::JointTrajectoryControllerStateConstPtr &msg);
    };
    bool determine_contact(Joint_Feedback &feedback);
  };
}

#endif // Robust_Grasp_H
