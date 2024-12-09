import cv2
import os
from pathlib import Path
import numpy as np
from PIL import Image
from utils.cos_util import COSUtil
from utils.video_clip_util import VideoClipUtil
import uuid
from urllib.parse import urlparse
import logging
import ffmpeg

'''
sudo apt update
sudo apt install ffmpeg
'''
class VideoHandler:
    def __init__(self):
        self.cos_util = COSUtil()
        self.video_clip_util = VideoClipUtil()
        self.project_dir = os.path.join(os.getcwd(), 'temp')
        self.base_cos_url = 'https://seeming-1322557366.cos.ap-chongqing.myqcloud.com'
        self.logger = logging.getLogger(__name__)

    
    '''
    处理关键帧
    '''
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
                frame_url = f"{self.base_cos_url}/{cos_path}"
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

    
            

    '''
    处理每一帧
    '''
    def process_all_frames(self, project_no, video_url, local_video_path=None, interval_seconds=1):
        """按固定时间间隔提取视频帧
        Args:
            project_no: 项目编号
            video_url: 视频URL
            local_video_path: 视频本地路径（如果已有）
            interval_seconds: 采样间隔(秒)，默认1秒
        Returns:
            dict: 包含视频信息和帧信息的字典
                {
                    "local_video_path": str,  # 视频本地路径
                    "frames": list[dict],     # 帧信息列表，每个元素包含local_frame_path和timestamp
                    "video_info": dict        # 视频基本信息（fps, duration等）
                }
        """
        try:
            if local_video_path is None:
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

            # 计算采样时间点（秒）
            sample_timestamps = range(0, duration, interval_seconds)
            
            frames = []
            
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
                
                Image.fromarray(frame_rgb).save(frame_path, quality=60)
                
                frames.append({
                    "local_frame_path": frame_path,
                    "timestamp": timestamp
                })
            
            cap.release()
            
            return {
                "local_video_path": local_video_path,
                "frames": frames,
                "video_info": {
                    "fps": fps,
                    "duration": duration,
                    "total_frames": total_frames
                }
            }

        except Exception as e:
            self.logger.error(f"Error processing frames: {str(e)}")
            return None


    
    '''
    合成视频
    '''
    def synthesize_video(self, selected_clips_directory, project_no):
        """
        Synthesize video clips based on selected segments and merge them.
        
        Args:
            materials (list): List of video materials with URLs
            selected_clips_directory (list): List of dictionaries containing video URLs and their selected segments
                Format: [{'video_url': str, 'segments': list[dict]}]
            project_no (str): Project identifier
            
        Returns:
            tuple: (merged_video_path, all_clip_paths, temp_files)
                - merged_video_path: Path to the final merged video
                - all_clip_paths: List of paths to individual clips
                - temp_files: List of temporary files created
                
        Raises:
            ValueError: If no valid clips could be generated or merged
        """
        clip_info_list = []

        try:
            # Generate video clips based on selected segments
            '''
            可以接受的clip的片段，等待合成
            '''
            if selected_clips_directory:
                for video_clips in selected_clips_directory:
                    video_url = video_clips['video_url']
                    segments = video_clips['segments']
                    
                    if not segments:
                        self.logger.warning(f"No segments selected for video: {video_url}")
                        continue
                        
                    '''
                    download origin videos
                    '''
                    video_local_path = self.download_video_from_cos(video_url, project_no)
                    self.logger.info(f"local video path: {video_local_path}")
                    if not video_local_path:
                        raise ValueError(f"Failed to download video from {video_url}")
                    
                    self.logger.info(f"Generating clips from video: {video_url}")
                    self.logger.info(f"Using clip segments: {segments}")
                    
                    # Generate clips using the selected segments
                    clip_paths = self.video_clip_util.generate_video_clips(
                        project_no,
                        segments,  # Format expected by generate_video_clips
                        video_local_path
                    )
                    
                    if clip_paths:
                        # 获取每个片段的时长并记录
                        for clip_path in clip_paths:
                            try:
                                # 使用ffmpeg获取视频时长
                                probe = ffmpeg.probe(clip_path)
                                duration = float(probe['streams'][0]['duration'])  # 使用 float 而不是 int
                                rounded_duration = round(duration)  # 四舍五入到最近的整数
                                
                                clip_info = {
                                    'path': clip_path,
                                    'duration': rounded_duration
                                }
                                clip_info_list.append(clip_info)
                            except Exception as e:
                                self.logger.error(f"Error getting duration for clip {clip_path}: {str(e)}")
                                # 如果获取时长失败，仍然添加路径但时长为None
                                clip_info_list.append({
                                    'path': clip_path,
                                    'duration': None
                                })
                        self.logger.info(f"Generated {len(clip_paths)} clips")
                    else:
                        self.logger.warning(f"No clips generated for video: {video_url}")

            if not clip_info_list:
                raise ValueError("No valid clips were generated from any video")

            # 获取所有clip路径用于合并
            all_clip_paths = [info['path'] for info in clip_info_list]

            # Merge the generated clips
            raw_merged_filename = f"{project_no}_raw_merged.mp4"
            raw_merged_video_path = os.path.join(self.project_dir, project_no, raw_merged_filename)
            
            raw_merged_video_path = self.video_clip_util.merge_video_clips(all_clip_paths, raw_merged_video_path)
            if not raw_merged_video_path:
                raise ValueError("Failed to merge video clips")

            return raw_merged_video_path, clip_info_list

        except Exception as e:
            self.logger.error(f"Error synthesizing video: {str(e)}")
            raise  # Re-raise the exception instead of returning None



    '''
    下载视频
    '''
    def download_video_from_cos(self, cos_video_url, project_no):
        """Download video from COS to local storage"""
        try:
            parsed_url = urlparse(cos_video_url)
            if not parsed_url.path:
                raise ValueError("Invalid COS URL: no path found: {parsed_url}")

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
        


    '''
    上传切片
    '''
    def upload_clips_to_cos(self, clip_paths, project_no):
        items = []
        failed_uploads = []
        
        for clip_path in clip_paths:
            try:
                # Generate a UUID for this upload
                upload_uuid = str(uuid.uuid4())
                
                # Create the COS path with project number, UUID, and 'clips' folder
                filename = os.path.basename(clip_path)
                remote_cos_clip_path = f"{project_no}/clips/{upload_uuid}_{filename}"
                
                # Upload the clip to COS
                self.cos_util.upload_file(
                    bucket=self.cos_util.bucket_name,
                    local_file_path=clip_path,
                    cos_file_path=remote_cos_clip_path
                )
                
                item = {
                    "url": f"{self.base_cos_url}/{remote_cos_clip_path}",
                }
                items.append(item)
                
                self.logger.info(f"Uploaded clip to COS: {remote_cos_clip_path}")
            except Exception as e:
                self.logger.error(f"Error uploading clip {clip_path} to COS: {str(e)}")
                failed_uploads.append({"path": clip_path, "error": str(e)})
                continue
        
        if failed_uploads:
            self.logger.error(f"Failed to upload {len(failed_uploads)} clips: {failed_uploads}")
            if not items:
                raise Exception(f"All uploads failed: {failed_uploads}")
        
        return items
    

    
    


    '''
    合并多个输入视频
    '''
    def merge_input_videos(self, video_urls, project_no):
        temp_files = []  # Track files to clean up
        
        try:
            project_dir = os.path.join(self.project_dir, project_no)
            os.makedirs(project_dir, exist_ok=True)
            
            # Download all videos and process them to have consistent parameters
            processed_video_paths = []
            for video_url in video_urls:
                local_path = self.download_video_from_cos(video_url, project_no)
                if not local_path:
                    raise ValueError(f"Failed to download video from {video_url}")
                temp_files.append(local_path)
                
                # Create processed video with consistent parameters
                processed_filename = f"processed_{os.path.basename(local_path)}"
                processed_path = os.path.join(project_dir, processed_filename)
                
                # Process each video to have consistent parameters
                try:
                    stream = (
                        ffmpeg
                        .input(local_path)
                        .filter('scale', 1080, 1920, force_original_aspect_ratio='decrease')
                        .filter('pad', 1080, 1920, '(ow-iw)/2', '(oh-ih)/2')
                        .output(processed_path, 
                            vcodec='libx264',
                            acodec='aac',
                            preset='medium',
                            crf=23)
                        .overwrite_output()
                    )
                    stream.run(capture_stderr=True)
                    processed_video_paths.append(processed_path)
                    temp_files.append(processed_path)
                except ffmpeg.Error as e:
                    if e.stderr:
                        self.logger.error(f"FFmpeg error processing video: {e.stderr.decode()}")
                    raise ValueError("Failed to process video")
                    
            self.logger.info(f"Processed {len(processed_video_paths)} videos successfully")
            
            # Create a concat file
            concat_file_path = os.path.join(project_dir, 'concat.txt')
            with open(concat_file_path, 'w') as f:
                for video_path in processed_video_paths:
                    f.write(f"file '{video_path}'\n")
            temp_files.append(concat_file_path)
            
            # Create output path for merged video
            merged_filename = f"{project_no}_merged_{uuid.uuid4()}.mp4"
            local_merged_video_path = os.path.join(project_dir, merged_filename)
            
            # Merge videos using concat demuxer
            try:
                stream = (
                    ffmpeg
                    .input(concat_file_path, format='concat', safe=0)
                    .output(local_merged_video_path, 
                        c='copy')  # Use copy codec for faster processing
                    .overwrite_output()
                )
                stream.run(capture_stderr=True)
            except ffmpeg.Error as e:
                if e.stderr:
                    self.logger.error(f"FFmpeg error merging videos: {e.stderr.decode()}")
                raise ValueError("Failed to merge videos using FFmpeg")
            
            if not os.path.exists(local_merged_video_path):
                raise FileNotFoundError("Merged video file was not created")
                
            self.logger.info(f"Successfully merged videos to: {local_merged_video_path}")
            return {
                "local_merged_video_path": local_merged_video_path,
                "project_no": project_no
            }
            
        except Exception as e:
            self.logger.error(f"Error merging input videos: {str(e)}")
            return None
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if temp_file and os.path.exists(temp_file):
                        os.remove(temp_file)
                        self.logger.debug(f"Cleaned up temporary file: {temp_file}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to clean up file {temp_file}: {str(cleanup_error)}")
        
    def upload_video_to_cos(self, local_video_path, project_no):
        """Upload merged video to COS
        
        Args:
            video_path (str): Local path to the video file
            project_no (str): Project identifier
            
        Returns:
            str: Public URL of the uploaded video
        """
        try:
            if not os.path.exists(local_video_path):
                raise FileNotFoundError(f"Local Video file not found: {local_video_path}")
            
            # Create the COS path with project number, UUID, and 'merged' folder
            filename = os.path.basename(local_video_path)
            remote_cos_path = f"{project_no}/merged/{str(uuid.uuid4())}_{filename}"
            
            self.logger.info(f"Starting upload of {local_video_path} to COS...")
            
            # Upload the video to COS
            self.cos_util.upload_file(
                bucket=self.cos_util.bucket_name,
                local_file_path=local_video_path,
                cos_file_path=remote_cos_path
            )
            
            # Generate and return the public URL
            video_url = f"{self.base_cos_url}/{remote_cos_path}"
            self.logger.info(f"Successfully uploaded merged video to COS: {remote_cos_path}")
            
            return {
                "video_url": video_url,
                "project_no": project_no
            }
            
        except Exception as e:
            self.logger.error(f"Error uploading merged video {local_video_path} to COS: {str(e)}")
            return None