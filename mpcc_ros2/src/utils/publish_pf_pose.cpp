#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/exceptions.h"

class PublishPfPose : public rclcpp::Node
{
public:
  PublishPfPose()
  : Node("publish_pf_pose"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // Parameters
    frame_id_ = this->declare_parameter<std::string>("frame_id", "map");
    child_frame_id_ = this->declare_parameter<std::string>("child_frame_id", "odom");

    // Publisher
    pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
      "/gmapping/odometry", 10);

    // Subscriber
    sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odometry/filtered", 10,
      std::bind(&PublishPfPose::odomCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "publish_pf_pose node started");
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    geometry_msgs::msg::TransformStamped odom_to_map;

    try {
      odom_to_map = tf_buffer_.lookupTransform(
        frame_id_,
        child_frame_id_,
        msg->header.stamp,
        rclcpp::Duration::from_seconds(1.0));
    }
    catch (tf2::TransformException & e) {
      RCLCPP_WARN(this->get_logger(),
        "[Particle Filter] Transform Lookup Exception: %s", e.what());
      return;
    }

    nav_msgs::msg::Odometry gmapping_odom;
    gmapping_odom.header.frame_id = frame_id_;
    gmapping_odom.header.stamp = this->now();

    try {
      tf2::doTransform(msg->pose.pose, gmapping_odom.pose.pose, odom_to_map);
    }
    catch (tf2::TransformException & e) {
      RCLCPP_WARN(this->get_logger(),
        "[Particle Filter] Transform Exception: %s", e.what());
      return;
    }

    gmapping_odom.twist.twist = msg->twist.twist;

    pub_->publish(gmapping_odom);
  }

  // Members
  std::string frame_id_;
  std::string child_frame_id_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PublishPfPose>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
