#include <mpcc_ros2/mpcc_ros2.h>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MPCCROS>();
  node->init();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
