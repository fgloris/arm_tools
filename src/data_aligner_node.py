#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2
import bisect
import torch
from tqdm import tqdm

# ROS 2 相关
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge

# LeRobot 相关
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class RosToLeRobotConverter:
    def __init__(self, bag_path, repo_id, fps=20):
        self.bag_path = bag_path
        self.repo_id = repo_id
        self.fps = fps
        self.bridge = CvBridge()
        
        # 话题定义
        self.topics = {
            'cam_left': '/camera_left/camera/color/image_rect_raw',
            'cam_right': '/camera_right/camera/color/image_rect_raw',
            'cam_high': '/camera_high/camera/color/image_rect_raw',
            'cam_low': '/camera_low/camera/color/image_rect_raw',
            'arm_left': '/motion_target/target_joint_state_arm_left',
            'arm_right': '/motion_target/target_joint_state_arm_right',
            'grip_left': '/motion_target/target_position_gripper_left',
            'grip_right': '/motion_target/target_position_gripper_right'
        }

        # 内存缓存索引
        self.state_buffer = {topic: [] for topic in self.topics.values() if 'camera' not in topic}
        self.state_times = {topic: [] for topic in self.topics.values() if 'camera' not in topic}
        self.image_buffer = {topic: [] for topic in self.topics.values() if 'camera' in topic}
        self.image_times = {topic: [] for topic in self.topics.values() if 'camera' in topic}
        self.topic_types = {}

    def read_bag(self):
        """流式索引 Bag 消息"""
        print(f"正在扫描 Bag: {self.bag_path}")
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='mcap')
        reader.open(storage_options, rosbag2_py.ConverterOptions('', ''))

        self.topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

        while reader.has_next():
            (topic, data, t_nanos) = reader.read_next()
            if topic not in self.topics.values(): continue
                
            msg_type = get_message(self.topic_types[topic])
            msg = deserialize_message(data, msg_type)
            t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if hasattr(msg, 'header') else t_nanos * 1e-9
            
            if 'camera' in topic:
                self.image_times[topic].append(t_sec)
                self.image_buffer[topic].append(data) # 存二进制字节，节省内存
            else:
                self.state_times[topic].append(t_sec)
                self.state_buffer[topic].append(msg)
        print("索引构建完成。")

    def _get_closest_after(self, times_list, buffer_list, target_time, is_image=False):
        idx = bisect.bisect_left(times_list, target_time)
        if idx < len(times_list):
            if is_image:
                msg = deserialize_message(buffer_list[idx], get_message(self.topic_types[self.topics['cam_high']]))
                img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                # LeRobot 期望图像为 (C, H, W)，此处先保持 (H, W, C) 后面由 dataset 处理或手动转
                return times_list[idx], cv2.resize(img, (640, 480))
            return times_list[idx], buffer_list[idx]
        return None, None

    def _format_qpos(self, arm_l, arm_r, grip_l, grip_r):
        l_pos = list(arm_l.position)
        r_pos = list(arm_r.position)
        l_g = grip_l.data if hasattr(grip_l, 'data') else grip_l.position[0]
        r_g = grip_r.data if hasattr(grip_r, 'data') else grip_r.position[0]
        return torch.tensor(l_pos + [l_g] + r_pos + [r_g], dtype=torch.float32)

    def run(self, task_name="aloha_task"):
        # 1. 初始化 LeRobot 数据集
        motors = ["L_w","L_s","L_e","L_f","L_wa","L_wr","L_g", "R_w","R_s","R_e","R_f","R_wa","R_wr","R_g"]
        features = {
            "observation.state": {"dtype": "float32", "shape": (14,), "names": motors},
            "action": {"dtype": "float32", "shape": (14,), "names": motors},
        }
        for cam in ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]:
            features[f"observation.images.{cam}"] = {"dtype": "video", "shape": (3, 480, 640), "names": ["c", "h", "w"]}

        dataset = LeRobotDataset.create(repo_id=self.repo_id, fps=self.fps, features=features, use_videos=True)

        # 2. 对齐逻辑
        start_time = max([t[0] for t in self.image_times.values() if t] + [t[0] for t in self.state_times.values() if t])
        end_time = min([t[-1] for t in self.image_times.values() if t] + [t[-1] for t in self.state_times.values() if t])
        
        dt = 1.0 / self.fps
        current_t = start_time
        
        # 缓存用于计算 Action (Next Qpos)
        temp_frames = []

        print("正在对齐并载入 LeRobot 格式...")
        while current_t < end_time:
            # 采样图像
            step_imgs = {}
            img_ts = []
            for key in ['cam_left', 'cam_right', 'cam_high', 'cam_low']:
                t_i, img = self._get_closest_after(self.image_times[self.topics[key]], self.image_buffer[self.topics[key]], current_t, True)
                if img is None: break
                step_imgs[key] = torch.from_numpy(img).permute(2, 0, 1) # HWC -> CHW
                img_ts.append(t_i)
            
            if len(step_imgs) < 4: break

            # 采样状态
            max_img_t = max(img_ts)
            states = {}
            for key in ['arm_left', 'arm_right', 'grip_left', 'grip_right']:
                _, msg = self._get_closest_after(self.state_times[self.topics[key]], self.state_buffer[self.topics[key]], max_img_t)
                if msg is None: break
                states[key] = msg
            
            if len(states) < 4: break

            qpos = self._format_qpos(states['arm_left'], states['arm_right'], states['grip_left'], states['grip_right'])
            
            frame_data = {
                "observation.state": qpos,
                "observation.images.cam_high": step_imgs['cam_high'],
                "observation.images.cam_low": step_imgs['cam_low'],
                "observation.images.cam_left_wrist": step_imgs['cam_left'],
                "observation.images.cam_right_wrist": step_imgs['cam_right'],
            }
            temp_frames.append(frame_data)
            current_t += dt

        # 3. 写入数据并计算 Action (Action_t = State_t+1)
        for i in tqdm(range(len(temp_frames))):
            next_idx = min(i + 1, len(temp_frames) - 1)
            temp_frames[i]["action"] = temp_frames[next_idx]["observation.state"]
            dataset.add_frame(temp_frames[i])

        dataset.save_episode(task=task_name)
        dataset.consolidate()
        print(f"数据集已保存至: {dataset.root}")

if __name__ == "__main__":
    bag_path = sys.argv[1].rstrip('/')
    repo_id = f"outputs/{os.path.basename(bag_path)}" # 本地存储路径
    converter = RosToLeRobotConverter(bag_path, repo_id)
    converter.read_bag()
    converter.run()