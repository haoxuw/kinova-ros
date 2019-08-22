#include <robust_grasp.h>
#include <ros/console.h>

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <tf_conversions/tf_eigen.h>
#include <thread>
#include <string>
#include <queue>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>

#include <control_msgs/JointTrajectoryControllerState.h>

#include "script_reader.h"

const double FINGER_MAX = 6400;

using namespace kinova;
using namespace std;

//#define HOME_POSE 0.0, -0.19, 0.25, 2.663, 0, 0
#define HOME_POSE 0, -0.2, 0.7, M_PI/2, 0, 0
#define INITIAL_CAMERA_POSE_X 0
#define INITIAL_CAMERA_POSE_Y -0.5
#define INITIAL_CAMERA_POSE_Z 0.977
#define WORKSPACE_LIMITS 1

static void rgbdCallback(RobustGrasp::Image_Buffer& ib, const sensor_msgs::CameraInfoConstPtr& ci, const sensor_msgs::ImageConstPtr& de, const sensor_msgs::ImageConstPtr& co) {
  ib.mutlock.lock();
  ib.camera_info = *ci;
  ib.color_msg = *co;
  ib.depth_msg = *de;
  ib.id ++;
  ib.mutlock.unlock();
}

RobustGrasp::RobustGrasp(ros::NodeHandle &nh):nh_(nh) {
  image_transport::ImageTransport it(nh_);

  camera_pose_x = INITIAL_CAMERA_POSE_X;
  camera_pose_y = INITIAL_CAMERA_POSE_Y;
  camera_pose_z = INITIAL_CAMERA_POSE_Z;

  nh_.param<std::string>("/robot_type",robot_type_,"j2s7s300");
  nh_.param<bool>("/robot_connected",robot_connected_,true);

  // Before we can load the planner, we need two objects, a RobotModel and a PlanningScene.
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model_ = robot_model_loader.getModel();

  // construct a `PlanningScene` that maintains the state of the world (including the robot).
  planning_scene_.reset(new planning_scene::PlanningScene(robot_model_));
  planning_scene_monitor_.reset(new planning_scene_monitor::PlanningSceneMonitor("robot_description"));

  //    //  every time need retrieve current robot state, do the following.
  //    robot_state::RobotState& robot_state = planning_scene_->getCurrentStateNonConst();
  //    const robot_state::JointModelGroup *joint_model_group = robot_state.getJointModelGroup("arm");

  group_ = new moveit::planning_interface::MoveGroup("arm");
  gripper_group_ = new moveit::planning_interface::MoveGroup("gripper");

  group_->setEndEffectorLink(robot_type_ + "_end_effector");

  finger_client_ = new actionlib::SimpleActionClient<kinova_msgs::SetFingersPositionAction>
    ("/" + robot_type_ + "_driver/fingers_action/finger_positions", false);
  while(robot_connected_ && !finger_client_->waitForServer(ros::Duration(5.0))){
    ROS_INFO("Waiting for the finger action server to come up");
  }

  pub_co_ = nh_.advertise<moveit_msgs::CollisionObject>("/collision_object", 10);
  pub_aco_ = nh_.advertise<moveit_msgs::AttachedCollisionObject>("/attached_collision_object", 10);
  pub_planning_scene_diff_ = nh_.advertise<moveit_msgs::PlanningScene>("planning_scene", 1);

  int arm_joint_num = robot_type_[3]-'0';
  joint_names_.resize(arm_joint_num);
  joint_values_.resize(joint_names_.size());
  for (uint i = 0; i<joint_names_.size(); i++)
    {
      joint_names_[i] = robot_type_ + "_joint_" + boost::lexical_cast<std::string>(i+1);
    }


  // set pre-defined joint and pose values.
  //define_cartesian_pose();
  //define_joint_values();
  clear_workscene();
  build_workscene();
  max_attempts = 5;
  geometry_msgs::PoseStamped spose = generate_pose(HOME_POSE);
  move_arm(spose);
  move_gripper("open");
}

RobustGrasp::~RobustGrasp()
{
  // shut down pub and subs
  //sub_joint_.shutdown();
  //sub_pose_.shutdown();
  pub_co_.shutdown();
  pub_aco_.shutdown();
  pub_planning_scene_diff_.shutdown();

  // release memory
  delete group_;
  delete gripper_group_;
  delete finger_client_;
}


void RobustGrasp::get_current_state(const sensor_msgs::JointStateConstPtr &msg) {
  boost::mutex::scoped_lock lock(mutex_state_);
  current_state_ = *msg;
}

void RobustGrasp::get_current_pose(const geometry_msgs::PoseStampedConstPtr &msg) {
  boost::mutex::scoped_lock lock(mutex_pose_);
  current_pose_ = *msg;
}

void RobustGrasp::clear_workscene()
{
  while(pub_planning_scene_diff_.getNumSubscribers() < 1) {
    ros::WallDuration(0.5).sleep();
  }
  moveit_msgs::CollisionObject collision_object;
  moveit_msgs::PlanningScene planning_scene_msg;
  collision_object.id = "surrounding";
  collision_object.operation = moveit_msgs::CollisionObject::REMOVE;
  pub_co_.publish(collision_object);
  planning_scene_msg.world.collision_objects.clear();
  planning_scene_msg.world.collision_objects.push_back(collision_object);

  planning_scene_msg.is_diff = true;
  pub_planning_scene_diff_.publish(planning_scene_msg);

  std::cout << "Surrounding scene removed" << std::endl;
}

#define THICKNESS 0.01
void RobustGrasp::build_workscene() {
  
  while(pub_planning_scene_diff_.getNumSubscribers() < 1) {
    ros::WallDuration(0.5).sleep();
  }
  //moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  moveit_msgs::CollisionObject collision_object;
  moveit_msgs::PlanningScene planning_scene_msg;
  collision_object.header.frame_id = "root";
  collision_object.id = "surrounding";

  shape_msgs::SolidPrimitive box;
  box.type = box.BOX;
  box.dimensions.resize(3);
  box.dimensions[0] = 2;
  box.dimensions[1] = 2;
  box.dimensions[2] = THICKNESS;
  geometry_msgs::Pose box_pose;
  box_pose.orientation.x = 0.0;
  box_pose.orientation.y = 0.0;
  box_pose.orientation.z = 0.0;
  box_pose.orientation.w = 1.0;
  box_pose.position.x = 0.0;
  box_pose.position.y = 0.0;
  box_pose.position.z = -THICKNESS/2.0;
  collision_object.primitives.push_back(box);
  collision_object.primitive_poses.push_back(box_pose);

  box.dimensions[0] = 0.3;
  box.dimensions[1] = 0.3;
  box.dimensions[2] = 0.3;
  box_pose.position.y = camera_pose_y;
  box_pose.position.z = camera_pose_z + 0.3/2; //CAMERA_HEIGHT is the bot of the camera, so we move up half it's box size
  collision_object.primitives.push_back(box);
  collision_object.primitive_poses.push_back(box_pose);
  
  box.dimensions[0] = THICKNESS;
  box.dimensions[1] = 2;
  box.dimensions[2] = 2;
  box_pose.position.x = -WORKSPACE_LIMITS;
  box_pose.position.y = 0;
  box_pose.position.z = 1;
  collision_object.primitives.push_back(box);
  collision_object.primitive_poses.push_back(box_pose);
  
  box.dimensions[0] = THICKNESS;
  box.dimensions[1] = 2;
  box.dimensions[2] = 2;
  box_pose.position.x = WORKSPACE_LIMITS;
  box_pose.position.y = 0;
  box_pose.position.z = 1;
  collision_object.primitives.push_back(box);
  collision_object.primitive_poses.push_back(box_pose);
  
  collision_object.operation = collision_object.ADD;

  planning_scene_msg.world.collision_objects.push_back(collision_object);
  planning_scene_msg.is_diff = true;
  pub_planning_scene_diff_.publish(planning_scene_msg);

  std::cout << "Surrounding scene created" << std::endl;
}

geometry_msgs::PoseStamped RobustGrasp::generate_pose(double x, double y, double z, double roll, double pitch, double yaw) {
  geometry_msgs::PoseStamped PS;
  PS.header.frame_id = "root";
  PS.header.stamp = ros::Time::now();
  PS.pose.position.x = x;
  PS.pose.position.y = y;
  PS.pose.position.z = z;
  tf::Quaternion q = tf::createQuaternionFromRPY(roll, pitch, yaw);
  PS.pose.orientation.x = q.x();
  PS.pose.orientation.y = q.y();
  PS.pose.orientation.z = q.z();
  PS.pose.orientation.w = q.w();
        
  //ROS_INFO("ORIENTATION: (%f, %f, %f, %f)", PS.pose.orientation.x, PS.pose.orientation.y, PS.pose.orientation.z, PS.pose.orientation.w);
  return PS;
}

moveit::planning_interface::MoveItErrorCode RobustGrasp::move_arm(geometry_msgs::PoseStamped &poseStamped, int attempts) {
  ROS_INFO("move_arm -- moving to: (%f ,%f %f)", poseStamped.pose.position.x, poseStamped.pose.position.y, poseStamped.pose.position.z);
  cout << "\t Pose " << poseStamped <<endl;
  moveit::planning_interface::MoveGroup & group = *group_;
  group.setPoseTarget(poseStamped);
  if (attempts < 0)
    attempts = max_attempts;
  for(int i=0;i<attempts;i++) {
    ROS_INFO("move_arm -- attemp # %d", i);
    moveit::planning_interface::MoveGroup::Plan plan;
    for(int j=0;j<max_attempts;j++) {
      group.setPlanningTime(2);
      ROS_INFO("move_arm -- planning...");
      auto plan_rc = group.plan(plan);
      ROS_INFO("move_arm -- planned");
      if(plan_rc == moveit::planning_interface::MoveItErrorCode::SUCCESS) {
        ROS_INFO("move_arm -- executing move now");
        auto exec_rc = group.execute(plan);
        if(exec_rc == moveit::planning_interface::MoveItErrorCode::SUCCESS || exec_rc.val == 0 || exec_rc == moveit::planning_interface::MoveItErrorCode::CONTROL_FAILED /* return code 0 seems to be a bug in moveit, and CONTROL_FAILED==-4 is ok to ignore */) {
          ROS_INFO("move_arm -- Successful, rc = %d.",(int)exec_rc.val);
          return moveit::planning_interface::MoveItErrorCode::SUCCESS;
        }
        else {
          ROS_INFO("move_arm -- Plan was found but execution failed with rc=%d. Retrying...",(int)exec_rc.val);
          continue;
        }
      }
      else
        ROS_INFO("move_arm -- Planning failed with rc=%d. Retrying...",(int)plan_rc.val);
    }
    ROS_INFO("move_arm -- No trojectory solution found, abort.");
    return moveit::planning_interface::MoveItErrorCode::PLANNING_FAILED;
  }
  ROS_INFO("move_arm -- Execution failed %d times, abort.", max_attempts);
  // see error encoding in http://docs.ros.org/hydro/api/ric_mc/html/classmoveit__msgs_1_1MoveItErrorCodes.html
  return moveit::planning_interface::MoveItErrorCode::MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE;
}

void RobustGrasp::Joint_Feedback::handle_feedback(const control_msgs::JointTrajectoryControllerStateConstPtr &msg) {
  end = ros::Time::now();
  double value = fabs(msg->error.positions[0]) * (fabs(msg->error.positions[1]) + fabs(msg->error.positions[2]));
  errors.push_back(value);
  //ROS_INFO("msg:  %f %f %f interpreted as : %f \n", msg->error.positions[0],msg->error.positions[1],msg->error.positions[2], value);
}

bool RobustGrasp::determine_contact(RobustGrasp::Joint_Feedback &feedback) {
  // right now, consider contact if the last SPAN feedbacks has been exceeded BAR in THRESH times
  const int SPAN = 3;
  const double BAR = 0.0001;
  const int THRESH = 1;
  int reach_count = 0;
  std::queue<double> queue;
  double running = 0;
  for(int i=0;i<feedback.errors.size();i++) {
    queue.push(feedback.errors[i]);
    running += queue.back();
    if(queue.size() == SPAN) {
      if(running/SPAN > BAR) {
        reach_count++;
        ROS_INFO("REACHED at #%d : %f", i, running/SPAN);
      }
      //else
      //ROS_INFO("PASSED at #%d : %f", i, running/THRESH);

      running -= queue.front();
      queue.pop();
    }
  }
  if(feedback.errors.size() < SPAN) {
    ;
  }
  return (reach_count >= THRESH);
}

bool RobustGrasp::move_to_homepose() {
  moveit::planning_interface::MoveItErrorCode rc;
  //rc.val |= move_gripper("open").val;
  rc.val |= move_gripper("close").val;
  auto pose = generate_pose(HOME_POSE);
  rc.val |= move_arm(pose).val;
  return rc == moveit::planning_interface::MoveItErrorCode::SUCCESS;
}

/*
  return code bit status:

  2 arm movement failed
  4 contact during arm movement -- unintended collision

  0 contact only happened when gripper close action -- success
  1 no contact -- missed


  std::string topic;
  if (robot_connected_)
  topic = "/j2s7s300_driver/out/joint_state";
  else
  topic = "/j2s7s300/effort_finger_trajectory_controller/state";
  moveit::planning_interface::MoveItErrorCode rc;
  {
  RobustGrasp::Joint_Feedback feedback_during_arm_movement;
  ros::Subscriber sub = nh_.subscribe(topic, 1024, &RobustGrasp::Joint_Feedback::handle_feedback, &feedback_during_arm_movement);
  //only try one time, fail means poor planning
  rc.val = move_arm(grasp_pose,1);
  if ( rc.val == moveit::planning_interface::MoveItErrorCode::PLANNING_FAILED ) {
  grasp_status |= 8;
  }
  else {
  if ( rc.val != moveit::planning_interface::MoveItErrorCode::SUCCESS )
  grasp_status |= 2;
  if(determine_contact(feedback_during_arm_movement))
  grasp_status |= 4;
  }
  if(grasp_status != 0)
  return grasp_status;
  }
  {
  RobustGrasp::Joint_Feedback feedback_during_gripping;
  ros::Subscriber sub = nh_.subscribe(topic, 1024, &RobustGrasp::Joint_Feedback::handle_feedback, &feedback_during_gripping);
  rc.val |= move_gripper("close");
  if(!determine_contact(feedback_during_gripping))
  grasp_status |= 1;
  if(grasp_status != 0)
  return grasp_status;
  }
  if(if_comeback) {
  ros::WallDuration(wait).sleep();
  // temp workaround TODO: go back to last pose
  pregrasp_pose = generate_pose(HOME_POSE);
  rc.val |= move_gripper("open");
  rc.val |= move_arm(pregrasp_pose);
  }

*/
moveit::planning_interface::MoveItErrorCode RobustGrasp::move_gripper(std::string target, double finger_turn) {
  moveit::planning_interface::MoveItErrorCode rc = 0;
  if(robot_connected_ == false) {
    if (!target.compare("open")) {
      ROS_INFO("move_gripper -- opening gripper");
      rc.val |= gripper_group_->setNamedTarget("Open");
      rc.val |= gripper_group_->move().val;
    }
    else if (!target.compare("close")) {
      ROS_INFO("move_gripper -- closing gripper");
      rc.val |= gripper_group_->setNamedTarget("Close");
      rc.val |= gripper_group_->move().val;
    }
    ROS_INFO("move_gripper -- gripper action completed with rc=%d.",(int)rc.val);
    return rc;
  }

  if (!target.compare("open"))
    finger_turn = 0;
  else if (!target.compare("close"))
    finger_turn = FINGER_MAX;
  else if (!target.compare("half"))
    finger_turn = FINGER_MAX/2;
       
  if (finger_turn < 0) {
    finger_turn = 0.0;
  }
  else {
    finger_turn = std::min(finger_turn, FINGER_MAX);
  }

  kinova_msgs::SetFingersPositionGoal goal;
  goal.fingers.finger1 = finger_turn;
  goal.fingers.finger2 = goal.fingers.finger1;
  goal.fingers.finger3 = goal.fingers.finger1;
  finger_client_->sendGoal(goal);

  if (finger_client_->waitForResult(ros::Duration(5.0)))
    {
      finger_client_->getResult();
      return true;
    }
  else
    {
      finger_client_->cancelAllGoals();
      ROS_WARN_STREAM("The gripper action timed-out");
      return false;
    }
}


#include <stdio.h>
#include <unistd.h>
#include <termios.h>

static int _getch(void)
{
  int ch;
  struct termios oldt;
  struct termios newt;
  tcgetattr(STDIN_FILENO, &oldt); /*store old settings */
  newt = oldt; /* copy old settings to new settings */
  newt.c_lflag &= ~(ICANON | ECHO); /* make one change to old settings in new settings */
  tcsetattr(STDIN_FILENO, TCSANOW, &newt); /*apply the new settings immediatly */
  ch = getchar(); /* standard getchar call */
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt); /*reapply the old settings */
  return ch; /*return received char */
}

// This function will put a color and a depth image in the respected location
// The unit of depth will be converted to meter
bool printed = false;

bool RobustGrasp::Image_Buffer::get_new_msg(sensor_msgs::Image &depth_msg_dest, sensor_msgs::Image &color_msg_dest, sensor_msgs::CameraInfo &info_msg_dest, double & distance, bool convert_from_mm) {
  bool newer = false;
  mutlock.lock();
  if(last_id < id) {
    depth_msg_dest = depth_msg;
    color_msg_dest = color_msg;
    info_msg_dest = camera_info;
    last_id = id;
    newer = true;

    cv_bridge::CvImagePtr dimge_cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    cv::Mat & de = dimge_cv_ptr->image;
    for(int i=0;i<de.cols;i++) {
      for(int j=0;j<de.rows;j++) {
	de.at<float>(j,i) = de.at<float>(j,i)/1000.0;
	//std::cout<< " "  << de.at<float>(j,i) ;
      }
    }
    distance = cv::mean(dimge_cv_ptr->image).val[0];
    depth_msg_dest = *(dimge_cv_ptr->toImageMsg());

    //////////////////
    /*
      cv::Rect ROI(0, 0, 400, 400);
      cv::Mat temp = de(ROI);
      temp.copyTo(de);
      info_msg_dest.width = 400;
      info_msg_dest.height = 400;
      ////////////////////

      std::cout << "depth width: "<<de.cols<< " height: "<<de.rows<<std::endl;
      std::cout << "depth value:"<< std::endl ;
      std::cout<<"Updated distant from camera: "<<distance<<std::endl;
    */


    /////////////////
    /*
      cv_bridge::CvImagePtr cimge_cv_ptr = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat & co = cimge_cv_ptr->image;;
      cv::Rect ROI2(0, 0, 400, 400);
      temp = co(ROI2);
      temp.copyTo(co);
      color_msg_dest = *(cimge_cv_ptr->toImageMsg());
      std_msgs::Header h = color_msg_dest.header;
      cout<<h<<endl; //all at once
      cout<<h.stamp<<endl; //specific parts of it
      cout<<h.stamp.sec<<endl;
      cout<<h.stamp.nsec<<endl;
      cout<<h.seq<<endl;
    */
    ///////////////////////

    if(!printed) {
      printed = true;
      std::cout<<"depth encoding : "<<depth_msg_dest.encoding<<std::endl;
      std::cout<<"height : "<<depth_msg_dest.height<<std::endl;
      std::cout<<"width  : "<<depth_msg_dest.width<<std::endl;
      std::cout<<"distant from camera: "<<distance<<std::endl;
    }
  }
  else {
    ROS_INFO("No new image, the latest id is # %ld", last_id);
  }
  mutlock.unlock();
  return newer;
}

bool RobustGrasp::Image_Buffer::get_new(cv::Mat &de, cv::Mat &co) {
  bool newer = false;
  mutlock.lock();
  if(last_id < id) {
    cv_bridge::toCvCopy(color_msg, "bgr8")->image.copyTo(co);
    cv_bridge::toCvCopy(depth_msg, "32FC1")->image.copyTo(de);
    de = de*0.001; // to meter
    last_id = id;
    newer = true;
  }
  else {
    ROS_INFO("No new image, the latest id is # %ld", last_id);
  }
  mutlock.unlock();
  return newer;
}

bool RobustGrasp::Image_Buffer::write_color_and_depth_to(std::string filename) {
  cv::Mat depth, color;
  if(get_new(depth, color)) {
    cv::imwrite (filename + ".png", color);
    cv::FileStorage file(filename+".xml", cv::FileStorage::WRITE);
    file<<"depth_image"<<depth;
    file.release();
    return true;
  }
  return false;
}

int RobustGrasp::run_script() {
  Sreader sreader;
  moveit::planning_interface::MoveItErrorCode rc;
  rc.val |= move_gripper("close").val;

  //auto pose = generate_pose(HOME_POSE);
  vector<geometry_msgs::PoseStamped> script = sreader.parse("");
  for (auto &pose : script) {
    rc.val |= move_arm(pose).val;
    ros::WallDuration(0.2).sleep();
  }
  return rc != moveit::planning_interface::MoveItErrorCode::SUCCESS;
}

void RobustGrasp::listen_to_console() {
  RobustGrasp & grasper = *this;
  std::string command;
  cv::Mat de, co;
  bool visualize = false;

  while(true) {
    std::cin>>command;
    if(command == "hello") {
      ROS_INFO_STREAM("world.");
    }
    else if(command == "quit" || command == "shutdown") {
      ros::shutdown();
    }
    else if(command == "script" || command == "s") {
      run_script();
    }
    else if(command == "takeover" || command == "t") {
      ROS_INFO_STREAM("Entered manual mode:");
      char ch = ' ';
      double inc = 0.05;
      geometry_msgs::PoseStamped spose = generate_pose(HOME_POSE);
      bool exe = false;
      bool disp_img = false;
      bool vertical = false;
      while(ch != 'q') {
	ROS_INFO_STREAM("Awaiting for instruction:");
	ch = _getch();
	if(ch == 's') {
	  if(vertical)
	    spose = generate_pose(0,-0.15,0.5, M_PI, 0, 0);
	  else
	    spose = generate_pose(HOME_POSE);
	  vertical = !vertical;
	}
	else if(ch == 'w')
	  spose.pose.position.y -= inc;
	else if(ch == 'x')
	  spose.pose.position.y += inc;
	else if(ch == 'a')
	  spose.pose.position.x += inc;
	else if(ch == 'd')
	  spose.pose.position.x -= inc;
	else if(ch == 'z')
	  spose.pose.position.z -= inc;
	else if(ch == 'c')
	  spose.pose.position.z += inc;
	else if(ch == 'e')
	  exe = !exe;
	else if(ch == '1') {
	  gripper_group_->setNamedTarget("Close");
	  gripper_group_->move();
	}
	else if(ch == '2') {
	  gripper_group_->setNamedTarget("Open");
	  gripper_group_->move();
	}
	else if(ch == 'f') {
	  spose = grasper.getCurrentPose();
	  grasper.stop();
	}
	else if(ch == ']') {
	  clear_workscene();
	}
	else if(ch == '[') {
	  build_workscene();
	}
	else if(ch == 'v') {
	  disp_img = !disp_img;
	  if(disp_img) {
	    if(image_buffer.get_new(de,co)) {
	      cv::namedWindow("preview_img");
	      cv::startWindowThread();
	      cv::imshow("preview_img", de/1.5);
	      image_buffer.write_color_and_depth_to("image_snapshot");
	      cv::waitKey(0);
	      cv::destroyWindow("preview_img");

	      cv::namedWindow("preview_img");
	      cv::startWindowThread();
	      cv::imshow("preview_img", co);
	      cv::waitKey(0);
	      cv::destroyWindow("preview_img");
	    }
	    else 
	      disp_img = !disp_img;
	  }
	  continue;
	}
	else if(ch == 'b') {
	  if(visualize) {
	    visualize = false;
	    cv::destroyWindow("preview_img");
	  }
	  else {
	    visualize = true;
	  }
	}
	else {
	  continue;
	}
	ROS_INFO("New manual target: (%f %f %f)", spose.pose.position.x, spose.pose.position.y, spose.pose.position.z);
	if(exe)
	  grasper.move_arm(spose);
      }
      ROS_INFO("Exited manual mode.");
    }
  }
}

bool RobustGrasp::start_service() {
  return true;
}

/*
  header: 
  seq: 1636
  stamp: 
  secs: 716
  nsecs: 782000000
  frame_id: camera_link
  height: 480
  width: 640
  encoding: 32FC1
  is_bigendian: 0
  step: 2560
*/

int main(int argc, char **argv)
{
  ros::init(argc, argv, "robust_grasp_demo");
  ros::NodeHandle nh;
  ros::AsyncSpinner spinner(3);
  spinner.start();

  RobustGrasp grasper(nh);

  std::thread listener(&RobustGrasp::listen_to_console, &grasper);

  grasper.start_service();


  std::string camera_info_topic;
  std::string depth_topic;
  std::string color_topic;
  if (grasper.robot_connected_) {
    //sub_joint_ = nh_.subscribe<sensor_msgs::JointState>("/j2s7s300_driver/out/joint_state", 1, &RobustGrasp::get_current_state, this);
    //sub_pose_ = nh_.subscribe<geometry_msgs::PoseStamped>("/" + robot_type_ +"_driver/out/tool_pose", 1, &RobustGrasp::get_current_pose, this);
    //message_filters::Subscriber<sensor_msgs::CameraInfo> cisub(nh,"/kinect2/hd/camera_info",1);
    //message_filters::Subscriber<sensor_msgs::Image> dsub(nh,"/kinect2/hd/image_depth_rect",1);
    //message_filters::Subscriber<sensor_msgs::Image> csub(nh,"/kinect2/hd/image_color_rect",1);
    //message_filters::Subscriber<sensor_msgs::CameraInfo> cisub(nh,"/kinect/color/camera_info",1);
    //message_filters::Subscriber<sensor_msgs::Image> dsub(nh,"/kinect/depth/image_raw",1);
    //message_filters::Subscriber<sensor_msgs::Image> csub(nh,"/kinect/color/image_raw",1);
    //camera_info_topic = "/kinect2/cd/camera_info";
    //depth_topic = "/kinect2/cd/depth";
    //color_topic = "/kinect2/cd/color";
    camera_info_topic = "/kinect2/hd/camera_info";
    depth_topic = "/kinect2/hd/image_depth_rect";
    color_topic = "/kinect2/hd/image_color_rect";
  }
  else {
    camera_info_topic = "/kinect/color/camera_info";
    depth_topic = "/kinect/depth/image_raw";
    color_topic = "/kinect/color/image_raw";
  }

  // get image
  message_filters::Subscriber<sensor_msgs::CameraInfo> cisub(nh,camera_info_topic,1);
  message_filters::Subscriber<sensor_msgs::Image> dsub(nh,depth_topic,1);
  message_filters::Subscriber<sensor_msgs::Image> csub(nh,color_topic,1);
  int herz = 10;
  message_filters::TimeSynchronizer<sensor_msgs::CameraInfo,sensor_msgs::Image,sensor_msgs::Image> sync(cisub, dsub, csub, herz);
  sync.registerCallback(boost::bind(&rgbdCallback, boost::ref(grasper.image_buffer), _1, _2, _3));
  // get image <---

  ros::spin();
  ros::shutdown();
  return 0;
}
