#!/usr/bin/env python3
import os
import sys
import h5py
import numpy as np
import cv2
import bisect
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from cv_bridge import CvBridge
from tqdm import tqdm

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
        
        # 内存缓存
        # 状态数据：{topic: [(t, msg)]} -> 全部存入内存
        self.state_buffer = {topic: [] for topic in self.topics.values() if 'camera' not in topic}
        self.state_times = {topic: [] for topic in self.topics.values() if 'camera' not in topic}
        
        # 图像数据：{topic: [(t, raw_data)]} -> 暂存原始序列化二进制，减少解析开销
        self.image_buffer = {topic: [] for topic in self.topics.values() if 'camera' in topic}
        self.image_times = {topic: [] for topic in self.topics.values() if 'camera' in topic}
        
        self.topic_types = {}

    def read_bag(self):
        """流式读取 Bag，区分存储状态和图像"""
        print(f"正在扫描 Bag (索引模式): {self.bag_path}")
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)

        self.topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

        while reader.has_next():
            (topic, data, t_nanos) = reader.read_next()
            if topic not in self.topics.values():
                continue
                
            msg_type = get_message(self.topic_types[topic])
            msg = deserialize_message(data, msg_type)
            t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if hasattr(msg, 'header') else t_nanos * 1e-9
            
            if 'camera' in topic:
                # 图像：只存时间戳和原始数据(data是bytes类型，比解析后的cv2图像轻量)
                self.image_times[topic].append(t_sec)
                self.image_buffer[topic].append(data) # 延迟解析
            else:
                # 状态：存入解析后的消息
                self.state_times[topic].append(t_sec)
                self.state_buffer[topic].append(msg)
        print("索引构建完成。")

    def get_closest_after_binary(self, times_list, buffer_list, target_time, is_image=False):
        """使用二分查找 target_time 右侧第一个消息"""
        idx = bisect.bisect_left(times_list, target_time)
        if idx < len(times_list):
            t = times_list[idx]
            raw_data_or_msg = buffer_list[idx]
            
            if is_image:
                # 如果是图像，在此处才进行反序列化和格式转换
                msg_type = get_message(self.topic_types[self.topics['cam_high']]) # 假设类型一致
                msg = deserialize_message(raw_data_or_msg, msg_type)
                return t, self.bridge.imgmsg_to_cv2(msg, "rgb8")
            return t, raw_data_or_msg
        return None, None

    def process_qpos(self, arm_l, arm_r, grip_l, grip_r):
        return np.concatenate([
            arm_l.position, [grip_l.data if hasattr(grip_l, 'data') else grip_l.position[0]], 
            arm_r.position, [grip_r.data if hasattr(grip_r, 'data') else grip_r.position[0]]
        ]).astype(np.float32)

    def align_and_save(self):
        """20Hz 对齐逻辑 + 实时 HDF5 写入"""
        # 计算起止时间
        all_t_starts = [t[0] for t in self.image_times.values() if t] + [t[0] for t in self.state_times.values() if t]
        all_t_ends = [t[-1] for t in self.image_times.values() if t] + [t[-1] for t in self.state_times.values() if t]
        
        start_time = max(all_t_starts)
        end_time = min(all_t_ends)
        
        target_fps = 20
        dt = 1.0 / target_fps
        num_steps = int((end_time - start_time) / dt)
        
        print(f"开始对齐转换，预计步数: {num_steps}")

        with h5py.File(self.output_h5, 'w') as f:
            # 预创建数据集空间
            obs = f.create_group('observations')
            imgs_g = obs.create_group('images')
            
            # 创建 chunked dataset 以节省内存并支持实时写入
            dsets = {}
            for k in ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']:
                dsets[k] = imgs_g.create_dataset(k, shape=(num_steps, 480, 640, 3), 
                                                dtype=np.uint8, chunks=(1, 480, 640, 3), compression="gzip")
            
            dset_qpos = obs.create_dataset('qpos', shape=(num_steps, 14), dtype=np.float32)
            dset_action = f.create_dataset('action', shape=(num_steps, 14), dtype=np.float32)

            all_qpos = []
            current_t = start_time
            step_count = 0

            for _ in tqdm(range(num_steps)):
                # 1. 采样图像 (target_t 右侧第一帧)
                step_imgs = {}
                img_times = []
                valid_step = True
                
                for key in ['cam_left', 'cam_right', 'cam_high', 'cam_low']:
                    topic = self.topics[key]
                    t_img, img_cv2 = self.get_closest_after_binary(self.image_times[topic], self.image_buffer[topic], current_t, is_image=True)
                    if img_cv2 is None: 
                        valid_step = False; break
                    img_times.append(t_img)
                    step_imgs[key] = cv2.resize(img_cv2, (640, 480))

                if not valid_step: break
                
                # 2. 采样状态 (图像采样点右侧的第一帧)
                max_img_t = max(img_times)
                state_msgs = {}
                for key in ['arm_left', 'arm_right', 'grip_left', 'grip_right']:
                    topic = self.topics[key]
                    _, msg = self.get_closest_after_binary(self.state_times[topic], self.state_buffer[topic], max_img_t)
                    if msg is None:
                        valid_step = False; break
                    state_msgs[key] = msg
                
                if not valid_step: break

                # 3. 写入当前帧数据
                qpos = self.process_qpos(state_msgs['arm_left'], state_msgs['arm_right'], 
                                         state_msgs['grip_left'], state_msgs['grip_right'])
                
                dset_qpos[step_count] = qpos
                dsets['cam_high'][step_count] = step_imgs['cam_high']
                dsets['cam_low'][step_count] = step_imgs['cam_low']
                dsets['cam_left_wrist'][step_count] = step_imgs['cam_left']
                dsets['cam_right_wrist'][step_count] = step_imgs['cam_right']
                
                all_qpos.append(qpos)
                step_count += 1
                current_t += dt
            
            # 4. 生成 Action (偏移一位)
            if step_count > 1:
                actions = np.zeros((step_count, 14), dtype=np.float32)
                actions[:-1] = np.array(all_qpos[1:])
                actions[-1] = all_qpos[-1]
                dset_action[:step_count] = actions

        print(f"转换完成，文件保存至: {self.output_h5}")

if __name__ == "__main__":
    # 使用示例
    import sys
    bag_path = sys.argv[1] # "path/to/your/bag_folder"
    output_h5 = bag_path.split('/')[-1][8:]+'.hdf5'
    aligner = DataAligner(bag_path, output_h5)
    aligner.read_bag()
    aligner.align_and_save()