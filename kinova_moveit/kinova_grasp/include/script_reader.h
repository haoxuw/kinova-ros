#ifndef script_reader_H
#define script_reader_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>


using namespace std;
class Sreader {
 public:
  Sreader() {
    mInterval = 1.0;
  }
  
  static vector<geometry_msgs::PoseStamped> parse(string fname) {
    vector<geometry_msgs::PoseStamped> traj;
    
    fname = "/home/haoxuw/mcgill/kinova/tracked_results/e010_res.traj";
    fname = "/home/haoxuw/mcgill/kinova/fake_results/0.fake.traj";
    fstream fs(fname);
    string line;
    while (getline(fs, line)) {
      cout<<line<<endl;
      stringstream ss(line);
      double time;
      ss >> time;
      double x;
      ss >> x;
      double y;
      ss >> y;
      double z;
      ss >> z;
      double roll;
      ss >> roll;
      double pitch;
      ss >> pitch;
      double yaw;
      ss >> yaw;
      
      geometry_msgs::PoseStamped ps;
      ps.header.frame_id = "root";
      ros::Duration duration(time);
      ps.header.stamp = ros::Time::now() + duration;
      ps.pose.position.x = x;
      ps.pose.position.y = y;
      ps.pose.position.z = z;
      tf::Quaternion q = tf::createQuaternionFromRPY(roll, pitch, yaw);
      ps.pose.orientation.x = q.x();
      ps.pose.orientation.y = q.y();
      ps.pose.orientation.z = q.z();
      ps.pose.orientation.w = q.w();

      traj.push_back(ps);
    }
    return traj;
  }

  vector<geometry_msgs::PoseStamped> &get_traj() {
    return mTraj;
  }
  
 private:
  vector<geometry_msgs::PoseStamped> mTraj;
  double mInterval;
};

#endif
