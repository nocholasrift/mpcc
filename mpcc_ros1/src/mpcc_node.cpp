#include <mpcc_ros1/mpcc_ros.h>
#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "mpcc_ros");
  ros::NodeHandle nh;

  MPCCROS mpcc_ros(nh);
  mpcc_ros.init();

  ros::AsyncSpinner spinner(1);
  spinner.start();
  ros::waitForShutdown();

  return 0;
}
