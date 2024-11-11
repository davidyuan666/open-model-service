import cv2
import os
from pathlib import Path
import numpy as np
from PIL import Image
from utils.cos_util import COSUtil
import uuid
from urllib.parse import urlparse


class VideoHandler:
    def __init__(self):
        self.cos_util = COSUtil()

    
    def process_keyframes(self, project_no, video_url, desired_frames=20):
        """根据视频长度和期望帧数提取关键帧
        Args:
            project_no: 项目编号
            video_url: 视频URL
            desired_frames: 期望返回的关键帧数量
        Returns:
            list: 包含关键帧信息的列表，每个元素为dict包含frame_url和timestamp
        """
        try:
            # Download video from COS
            local_video_path = self.download_video_from_cos(video_url, project_no)
            if not local_video_path:
                raise ValueError("Failed to download video from COS")

            # Create output directory for frames
            frames_dir = os.path.join(os.getcwd(), 'temp', project_no, 'frames')
            os.makedirs(frames_dir, exist_ok=True)

            # Open video file
            cap = cv2.VideoCapture(local_video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = int(total_frames / fps)  # 视频总时长(秒)

            # 计算采样间隔(秒)，稍微密集一些采样
            sample_interval = max(1, int(duration / (desired_frames * 1.2)))
            
            # 计算实际要采样的时间点（秒）
            sample_timestamps = range(0, duration, sample_interval)
            
            keyframes = []
            
            for timestamp in sample_timestamps:
                # 计算对应的帧位置
                frame_pos = int(timestamp * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save frame locally
                frame_filename = f"frame_{uuid.uuid4()}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                
                # Save using PIL for better quality
                Image.fromarray(frame_rgb).save(frame_path, quality=95)
                
                # Upload frame to COS
                cos_path = f"{project_no}/frames/{frame_filename}"
                self.cos_util.upload_file(
                    self.cos_util.bucket_name,
                    frame_path,
                    cos_path
                )
                
                # Get public URL and add to results
                frame_url = f"https://seeming-1322557366.cos.ap-chongqing.myqcloud.com/{cos_path}"
                keyframes.append({
                    "frame_url": frame_url,
                    "timestamp": timestamp
                })
                
                # Clean up local frame file
                os.remove(frame_path)

            # Clean up
            cap.release()
            os.remove(local_video_path)
            
            # 如果采样得到的帧数超过需要的数量，进行均匀裁剪
            if len(keyframes) > desired_frames:
                indices = np.linspace(0, len(keyframes)-1, desired_frames, dtype=int)
                keyframes = [keyframes[i] for i in indices]
            
            return keyframes

        except Exception as e:
            print(f"Error processing keyframes: {str(e)}")
            raise

    
            

    def download_video_from_cos(self, cos_video_url, project_no):
        """Download video from COS to local storage"""
        try:
            parsed_url = urlparse(cos_video_url)
            if not parsed_url.path:
                raise ValueError("Invalid COS URL: no path found")

            object_key = parsed_url.path.lstrip('/')
            original_filename = os.path.basename(object_key)
            
            if not original_filename:
                raise ValueError("Invalid COS URL: no filename found")
            
            video_local_path = os.path.join(os.getcwd(), 'temp', project_no, original_filename)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(video_local_path), exist_ok=True)

            # Download the file
            self.cos_util.download_file(
                self.cos_util.bucket_name,
                object_key,
                video_local_path
            )
            
            if not os.path.exists(video_local_path):
                raise FileNotFoundError("File download failed")

            return video_local_path
            
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            return None