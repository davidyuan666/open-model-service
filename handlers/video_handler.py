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

    def extract_key_frames(self, video_url, project_no, frame_interval=1, max_frames=10):
        """
        Extract key frames from a video file
        
        Args:
            video_url (str): COS URL of the video file
            project_no (str): Project number
            frame_interval (int): Interval between frames in seconds
            max_frames (int): Maximum number of frames to extract
            
        Returns:
            list: List of frame URLs uploaded to COS
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
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps

            # Calculate frame extraction points
            frame_interval_frames = int(frame_interval * fps)
            frame_positions = np.linspace(0, frame_count-1, min(max_frames, int(duration/frame_interval)+1))
            
            frame_urls = []
            
            for frame_pos in frame_positions:
                # Set video position
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_pos))
                
                # Read frame
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
                
                # Get public URL
                frame_url = f"https://{self.cos_util.bucket_name}.cos.{self.cos_util.config.region}.myqcloud.com/{cos_path}"
                frame_urls.append(frame_url)
                
                # Clean up local frame file
                os.remove(frame_path)

            # Clean up
            cap.release()
            os.remove(local_video_path)
            
            return frame_urls

        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            return None

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