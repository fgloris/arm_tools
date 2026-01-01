#include <iostream>
#include <deque>
#include <mutex>
#include <rclcpp/clock.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

class TopicAnalyzer : public rclcpp::Node {
public:
TopicAnalyzer() : Node("topic_analyzer") {
  this->declare_parameter<std::string>("topic_name", "/camera_left/camera/color/image_rect_raw");
  this->declare_parameter<long>("time_window_size", 10);
  
  this->get_parameter("topic_name", target_topic_);
  this->get_parameter("time_window_size", time_window_size_);

  RCLCPP_INFO(this->get_logger(), "Analyzing topic: %s", target_topic_.c_str());

  // 设置 QoS：对于传感器数据，使用 SensorDataQoS (Best Effort) 避免阻塞
  auto qos = rclcpp::SensorDataQoS();

  subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
    target_topic_, qos,
    std::bind(&TopicAnalyzer::topic_callback, this, std::placeholders::_1));

  start_time_ = this->now();
  timer_ = this->create_wall_timer(
    std::chrono::seconds(1), std::bind(&TopicAnalyzer::log_statistics, this));
}

private:
void topic_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  auto header_stamp = rclcpp::Time(msg->header.stamp);
  auto now = this->now();

  std::lock_guard<std::mutex> lock(mutex_);

  latencies_.push_back((now - header_stamp).seconds());
  arrival_times_.push_back(now);
  header_stamps_.push_back(header_stamp);
}

void log_statistics() {
  std::unique_lock<std::mutex> lock(mutex_);
  auto now = this->now();
  if ((now - start_time_).seconds() < time_window_size_){
    lock.unlock();
    RCLCPP_WARN(this->get_logger(), "Filling window, please wait...");
    return;
  }

  while (!arrival_times_.empty() && (now - arrival_times_.front()).seconds() > time_window_size_) {
    latencies_.pop_front();
    arrival_times_.pop_front();
    header_stamps_.pop_front();
  }
  double arrival_count = static_cast<double>(arrival_times_.size()) / static_cast<double>(time_window_size_);
  
  if (arrival_times_.size() == 0) {
    lock.unlock();
    RCLCPP_WARN(this->get_logger(), "No messages received on: %s", target_topic_.c_str());
    return;
  }

  double avg_delta = 0.0;
  if (arrival_count > 1){
    const auto [min, max] = std::minmax_element(begin(header_stamps_), end(header_stamps_));
    avg_delta = (*max - *min).seconds() / (header_stamps_.size() - 1);
  }

  double avg_latency = std::accumulate(latencies_.begin(), latencies_.end(), 0.0) / latencies_.size();
  double max_latency = *std::max_element(latencies_.begin(), latencies_.end());

  lock.unlock();
  printf("\033[2J\033[H");
  std::cout << "========================================" << std::endl;
  std::cout << " Analyzing Topic: " << this->get_parameter("topic_name").as_string() << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  std::cout << " FPS (from arrival package): " << arrival_count << " Hz" << std::endl;
  std::cout << " Avg Header Delta:           " << avg_delta * 1000.0 << " ms" << std::endl;
  if (avg_delta > 0) {
  std::cout << " FPS (from header increment):" << 1.0 / avg_delta << " Hz" << std::endl;
  }
  std::cout << " Avg Latency:                " << avg_latency << std::endl;
  std::cout << " Max Latency:                " << max_latency << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << " (Press Ctrl+C to quit)" << std::endl;
}

rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
rclcpp::TimerBase::SharedPtr timer_;
std::deque<rclcpp::Time> arrival_times_;
std::deque<rclcpp::Time> header_stamps_;
std::deque<double> latencies_;
std::mutex mutex_;
rclcpp::Time start_time_;

std::string target_topic_;
unsigned long time_window_size_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TopicAnalyzer>());
  rclcpp::shutdown();
  return 0;
}