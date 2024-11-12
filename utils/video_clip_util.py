import tempfile
import uuid
import os
import json
from moviepy.editor import (
    VideoFileClip, 
    concatenate_videoclips,
)
import tempfile
import subprocess
import shutil
import logging


class VideoClipUtil:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        
    def check_ffmpeg(self):
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except:
            return False

    def ensure_dir(self,file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)



    def extract_audio(self, project_no, video_path):
        video = None
        try:
            video = VideoFileClip(video_path)
            if video.audio is None:
                print(f"Warning: The video at {video_path} does not have an audio track.")
                return None

            # 使用当前临时目录
            temp_dir = os.path.join(os.getcwd(),'temp',project_no)
            
            # 生成一个唯一的文件名，移除 .mp4 扩展名（如果存在）
            video_basename = os.path.basename(video_path)
            video_name_without_ext = os.path.splitext(video_basename)[0]
            audio_filename = f"audio_{video_name_without_ext}.wav"
            audio_path = os.path.join(temp_dir, audio_filename)
            
            if video.audio is not None:
                video.audio.write_audiofile(audio_path)
                return audio_path
            else:
                print(f"Warning: Unable to extract audio from {video_path}")
                return None
        except Exception as e:
            print(f"Error extracting audio from video {video_path}: {str(e)}")
            return None
        finally:
            if video is not None:
                video.close()
    


    '''
    合成视频
    '''
    def merge_video_clips(self, clip_paths, merged_video_path):
        """
        直接根据视频片段路径合并视频
        
        Args:
            clip_paths: 视频片段路径列表
            merged_video_path: 合并后的视频保存路径
            
        Returns:
            str: 合并后的视频路径，失败则返回None
        """
        if not clip_paths:
            raise ValueError("No clip paths provided")

        if not self.check_ffmpeg():
            raise RuntimeError("FFmpeg is not installed or not in the system PATH")

        temp_output_path = None
        video_clips = []

        try:
            # 确保输出目录存在
            self.ensure_dir(merged_video_path)

            # 创建临时输出文件
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_output_path = temp_output.name
            temp_output.close()

            # 加载所有视频片段
            for index, clip_path in enumerate(clip_paths):
                self.logger.info(f"Processing clip {index}: {clip_path}")
                
                if not os.path.exists(clip_path):
                    self.logger.error(f"Clip file not found: {clip_path}")
                    continue
                
                try:
                    video_clip = VideoFileClip(clip_path)
                    video_clips.append(video_clip)
                except Exception as e:
                    self.logger.error(f"Error loading clip {index} from path {clip_path}: {str(e)}")
                    continue

            if not video_clips:
                raise ValueError("No valid video clips to merge")

            # 合并视频片段
            self.logger.info(f"Merging {len(video_clips)} video clips")
            final_clip = concatenate_videoclips(video_clips, method="compose")

            # 写入临时文件
            self.logger.info(f"Writing merged video to temporary file: {temp_output_path}")
            final_clip.write_videofile(
                temp_output_path,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                fps=24
            )

            # 如果最终输出文件已存在，则删除
            if os.path.exists(merged_video_path):
                os.remove(merged_video_path)

            # 复制临时文件到最终位置
            shutil.copy2(temp_output_path, merged_video_path)

            self.logger.info(f"Video successfully merged and saved to: {merged_video_path}")
            return merged_video_path

        except Exception as e:
            self.logger.error(f"Error during video merging: {str(e)}", exc_info=True)
            return None

        finally:
            # 清理资源
            for clip in video_clips:
                try:
                    clip.close()
                except:
                    pass
                    
            if 'final_clip' in locals():
                try:
                    final_clip.close()
                except:
                    pass
                    
            if temp_output_path and os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except:
                    pass



    '''
    生成切片视频
    '''
    def generate_video_clips(self, project_no,clip_transcription, download_video_path):
        try:
            clip_paths = []

            print(f"Debug: clip_transcription type = {type(clip_transcription)}")
        
            
            if isinstance(clip_transcription, str):
                try:
                    # 尝试不同的编码方式解析JSON
                    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'shift-jis', 'euc-jp']
                    for encoding in encodings:
                        try:
                            # 如果是字符串，先编码再解码确保编码正确
                            decoded_str = clip_transcription.encode(encoding).decode(encoding)
                            clip_transcription = json.loads(decoded_str)
                            self.logger.info(f"Successfully parsed JSON with {encoding} encoding")
                            break
                        except (UnicodeError, json.JSONDecodeError):
                            continue
                    else:
                        raise ValueError("Failed to parse JSON with any known encoding")
                        
                except Exception as e:
                    self.logger.error(f"Failed to parse clip_transcription: {str(e)}")
                    self.logger.error(f"Original clip_transcription: {repr(clip_transcription)}")
                    raise ValueError(f"Failed to parse clip_transcription: {str(e)}")
                
            # 确保 clip_transcription 是字典类型
            if not isinstance(clip_transcription, dict):
                raise ValueError(f"clip_transcription must be a dictionary, but got {type(clip_transcription)}")
            
            # 获取转录列表
            transcription = clip_transcription.get('transcription', [])
            
            if len(transcription) == 0:
                print("No transcriptions found in clip_transcription")
                return clip_paths
            
            print(f"Debug: Processing video_path = {download_video_path}")

            video_name = os.path.basename(download_video_path)
            clips_dir = os.path.join(os.getcwd(),'temp',f'{project_no}','clips')
            os.makedirs(clips_dir, exist_ok=True)

                
            with VideoFileClip(download_video_path) as video:
                timestamps = clip_transcription['transcription']
                    
                print(f"Debug: timestamps for {os.path.basename(download_video_path)} = {json.dumps(timestamps, indent=2)}")
                    
                for i, timestamp in enumerate(timestamps):
                    if 'start' not in timestamp or 'end' not in timestamp:
                        print(f"Warning: Skipping invalid timestamp: {timestamp}")
                        continue
                        
                    start_time = timestamp['start']
                    end_time = timestamp['end']
                        
                    try:
                        clip = video.subclip(start_time, end_time)
                        if clip.duration == 0:
                            print(f"Warning: Skipping zero-duration clip for start={start_time}, end={end_time}")
                            continue
                            
                        clip_filename = f"{video_name}_clip_{i}_{uuid.uuid4()}.mp4"
                        clip_path = os.path.join(clips_dir, clip_filename)
                        clip.write_videofile(clip_path, codec="libx264", audio_codec="aac")
                        print(f"Saved individual clip: {clip_path}")
                            
                        clip_paths.append(clip_path)
                    except Exception as e:
                        print(f"Error creating subclip for start={start_time}, end={end_time}: {str(e)}")
                

            if not clip_paths:
                raise ValueError("No valid video clips were generated")

            print(f"Debug: Number of clips generated: {len(clip_paths)}")
            return clip_paths

        except Exception as e:
            print(f"Error generating video clips: {str(e)}")
            raise
    



    
    