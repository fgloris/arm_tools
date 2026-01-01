#!/usr/bin/env python3
import os
import h5py
import numpy as np
import cv2
from datetime import datetime
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from cv_bridge import CvBridge

class DataAligner:
    def __init__(self, bag_path, output_h5):
        self.bag_path = bag_path
        self.output_h5 = output_h5
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
        
        # 数据缓存: {topic: [(timestamp, msg), ...]}
        self.data_buffer = {topic: [] for topic in self.topics.values()}

    def read_bag(self):
        """读取整段 Bag 到内存缓存"""
        print(f"Reading bag: {self.bag_path}")
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)

        topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

        while reader.has_next():
            (topic, data, t_nanos) = reader.read_next()
            if topic in self.data_buffer:
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                # 使用 header.stamp 如果有，否则用接收时间
                t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if hasattr(msg, 'header') else t_nanos * 1e-9
                self.data_buffer[topic].append((t_sec, msg))
        print("Bag loaded.")

    def get_closest_after(self, topic, target_time):
        """寻找 target_time 之后的第一条消息 (No Interpolation)"""
        buffer = self.data_buffer[topic]
        for t, msg in buffer:
            if t >= target_time:
                return t, msg
        return None, None

    def process_qpos(self, arm_l, arm_r, grip_l, grip_r):
        """将左右臂和夹爪拼接成 Aloha 14D 向量"""
        # 假设顺序: left_arm(6) + left_grip(1) + right_arm(6) + right_grip(1)
        # 具体顺序需根据你的模型修改
        qpos = np.concatenate([
            arm_l.position, [grip_l.data], 
            arm_r.position, [grip_r.data]
        ]).astype(np.float32)
        return qpos

    def align_and_save(self):
        """执行 20Hz 对齐逻辑"""
        # 确定起止时间 (以所有话题都有数据为准)
        start_time = max([buf[0][0] for buf in self.data_buffer.values() if buf])
        end_time = min([buf[-1][0] for buf in self.data_buffer.values() if buf])
        
        target_fps = 20
        dt = 1.0 / target_fps
        
        all_qpos = []
        all_images = {k: [] for k in ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']}
        
        current_t = start_time
        while current_t < end_time:
            # 1. 采样图像 (target_t 右侧第一帧)
            imgs = {}
            img_times = []
            valid_step = True
            
            for key, topic in self.topics.items():
                if 'camera' in topic:
                    t_img, msg_img = self.get_closest_after(topic, current_t)
                    if msg_img is None: 
                        valid_step = False; break
                    img_times.append(t_img)
                    imgs[key] = self.bridge.imgmsg_to_cv2(msg_img, "rgb8")

            if not valid_step: break
            
            # 2. 采样状态 (图像采样点右侧的第一帧)
            max_img_t = max(img_times)
            state_msgs = {}
            for key in ['arm_left', 'arm_right', 'grip_left', 'grip_right']:
                _, msg = self.get_closest_after(self.topics[key], max_img_t)
                if msg is None:
                    valid_step = False; break
                state_msgs[key] = msg
            
            if not valid_step: break

            # 3. 存储
            qpos = self.process_qpos(state_msgs['arm_left'], state_msgs['arm_right'], 
                                     state_msgs['grip_left'], state_msgs['grip_right'])
            all_qpos.append(qpos)
            all_images['cam_high'].append(imgs['cam_high'])
            all_images['cam_low'].append(imgs['cam_low'])
            all_images['cam_left_wrist'].append(imgs['cam_left'])
            all_images['cam_right_wrist'].append(imgs['cam_right'])

            current_t += dt

        # 4. 写入 HDF5
        with h5py.File(self.output_h5, 'w') as f:
            obs = f.create_group('observations')
            imgs_group = obs.create_group('images')
            for k, v in all_images.items():
                imgs_group.create_dataset(k, data=np.array(v), chunks=(1, 480, 640, 3), compression="gzip")
            
            qpos_arr = np.array(all_qpos)
            obs.create_dataset('qpos', data=qpos_arr)
            
            # Action 是下一帧的 qpos (Aloha 标准)
            action = np.zeros_like(qpos_arr)
            action[:-1] = qpos_arr[1:]
            action[-1] = qpos_arr[-1]
            f.create_dataset('action', data=action)

        print(f"Successfully saved to {self.output_h5}")

if __name__ == "__main__":
    # 使用示例
    import sys
    bag_path = sys.argv[1] # "path/to/your/bag_folder"
    output_h5 = bag_path.split('/')[-1][8:]+'.hdf5'
    aligner = DataAligner(bag_path, output_h5)
    aligner.read_bag()
    aligner.align_and_save()