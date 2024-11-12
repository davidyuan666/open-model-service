import tempfile
import math
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm
from utils.llm_util import LLMUtil
import re
import uuid
import os
import json
from moviepy.editor import (
    VideoFileClip, 
    TextClip, 
    AudioFileClip,
    CompositeAudioClip,
    concatenate_videoclips,
    CompositeVideoClip,
    vfx
)
from moviepy.video.VideoClip import ImageClip
import tempfile
import subprocess
import shutil
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from typing import List
from pydantic import BaseModel, Field

class Transcription(BaseModel):
    """音频文本片段模型"""
    start: float = Field(..., description="开始时间（秒）")
    end: float = Field(..., description="结束时间（秒）")
    text: str = Field(..., description="文本内容")

class TranscriptionResponse(BaseModel):
    """音频文本响应模型"""
    transcription: List[Transcription] = Field(..., description="选中的音频文本片段列表")

    class Config:
        json_schema_extra = {
            "example": {
                "transcription": [
                    {
                        "start": 0.0,
                        "end": 11.0,
                        "text": "示例文本内容"
                    }
                ]
            }
        }


'''
sudo apt-get install fonts-wqy-zenhei fonts-wqy-microhei
'''
class VideoClipUtil:
    def __init__(self):
        self.llm_util = LLMUtil()
        self.logger = logging.getLogger(__name__)

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


    def analyze_image_by_blip(self, image_url, project_no):
        """
        通过BLIP API分析图片内容
        
        Args:
            image_url (str): 图片URL
            project_no (str): 项目编号
            
        Returns:
            dict: 包含分析结果的字典，格式如下：
                {
                    "description": str,  # 图片描述
                    "error": str,        # 错误信息（如果有）
                }
        """
        try:
            import requests
            
            # API endpoint
            api_url = "http://workspace.featurize.cn:60048/blip/generate_caption"
            
            # 准备请求数据
            payload = {
                "image_url": image_url,
                "project_no": project_no
            }
            
            # 发送POST请求
            response = requests.post(api_url, json=payload)
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json()
                return {
                    "description": result.get("caption", "").strip(),
                    "error": None
                }
            else:
                error_msg = f"API request failed with status code: {response.status_code}"
                self.logger.error(error_msg)
                return {
                    "description": "",
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"Error analyzing image with BLIP: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "description": "",
                "error": error_msg
            }
        



    def analyze_image(self, image_base64, lang='zh'):
        """
        分析图像内容并返回描述
        
        Args:
            image_base64 (str): Base64编码的图像数据
            lang (str): 语言代码，默认为中文
            
        Returns:
            dict: 包含分析结果的字典
        """
        try:
            # 定义不同语言的提示语
            prompts = {
                'zh': {
                    'main': "请分析这张图片，描述图片内容，方便剪辑，用简洁的语言描述。"
                },
                'en': {
                    'main': "Please analyze this image, describe its content for editing purposes, use concise language."
                },
                'ja': {
                    'main': "この画像を分析し、編集に適した内容を簡潔な言葉で説明してください。"
                }
            }

            # 获取对应语言的提示语
            prompt = prompts.get(lang, prompts['zh'])

            # 调用LLM进行图像分析
            description = self.llm_util.analyze_image_base64(
                prompt['main'],
                image_base64
            )

            return {
                "description": description.strip(),
                "language": lang
            }

        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}", exc_info=True)
            return {
                "description": "",
                "error": str(e)
            }



    def select_and_merge_segments(self, segments, prompt, preset):
        """
        选择并合并视频和音频分析的片段
        
        Args:
            segments (list): 包含视频和音频分析结果的片段列表
            prompt (str): 用户提示词
            preset (dict): 预设配置
            
        Returns:
            list: 合并后的片段列表，每个片段包含 start, end, text 信息
        """
        try:
            # 按时间戳排序所有片段
            segments.sort(key=lambda x: x['start'])
            lang = preset.get('narrationLang', 'zh')
            
            # 准备分析文本
            segments_text = "\n\n".join([
                f"开始时间点: {seg['start']}秒\n"
                f"结束时间点: {seg['end']}秒\n"
                f"内容: {seg['text']}\n"
                for seg in segments
            ])
            
            # 根据语言选择提示词
            if lang == 'en':
                analysis_prompt = f"""Based on the clip description, select relevant segments from the following clips. Return the selected segments in the original format as required JSON.

    Clips to process:
    {segments_text}

    Clip description: {prompt}

    Constraints:
    - Return format must match the input format exactly
    - Only return segments that match the requirements
    - Can select one or multiple segments
    - Each segment must include start time, end time, and content
    - All text must be in English
    - Do not add any extra explanations or modify the original data structure"""

            elif lang == 'ja':
                analysis_prompt = f"""クリップの説明に基づいて、以下のクリップから関連するセグメントを選択してください。選択したセグメントを元の形式で必要なJSONとして返してください。

    処理するクリップ:
    {segments_text}

    クリップの説明: {prompt}

    制約：
    - 返却フォーマットは入力フォーマットと完全に一致する必要があります
    - 要件に一致するセグメントのみを返してください
    - 1つまたは複数のセグメントを選択できます
    - 各セグメントは開始時間、終了時間、内容を含む必要があります
    - すべてのテキストは日本語である必要があります
    - 追加の説明を加えたり、元のデータ構造を変更したりしないでください"""

            else:  # default to Chinese
                analysis_prompt = f"""根据剪辑描述，从以下待剪辑片段中选择符合要求的片段。返回选中的片段，保持原始格式不变,按照要求的JSON格式返回。

    待剪辑片段:
    {segments_text}

    剪辑描述: {prompt}

    约束：
    - 返回格式必须与输入的"待剪辑片段"格式完全一致
    - 只返回你认为符合剪辑要求的片段
    - 可以选择一个或多个片段
    - 每个片段必须包含开始时间、结束时间和内容
    - 所有文本必须是中文
    - 不要添加任何额外解释或修改原始数据结构"""

            # 获取LLM的选择结果
            response = self.llm_util.native_structured_chat(analysis_prompt, TranscriptionResponse)
            
            if response and hasattr(response, 'parsed'):
                # 将Pydantic模型转换为字典
                result = response.parsed.model_dump()
                
                return result
            else:
                self.logger.error("Invalid response format received")
                return None

        except Exception as e:
            self.logger.error(f"Error in select_and_merge_segments: {str(e)}", exc_info=True)
            return None

    
    def create_text_frame(self, text, frame_size, fontsize=36, color=(255, 255, 255), position='bottom', subtitle_font=None):
        """创建包含文本的透明帧，支持中文和日文"""
        try:
            # 创建透明背景
            frame = np.zeros((frame_size[1], frame_size[0], 4), dtype=np.uint8)
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)
            
            # 检测文本语言
            def is_japanese(text):
                return any('\u3040' <= c <= '\u30ff' or '\u3400' <= c <= '\u4dbf' for c in text)
            
            is_jp = is_japanese(text)
            self.logger.info(f"Text language detection: {'Japanese' if is_jp else 'Other'}")
            
            subtitle_font_paths = {
                # 水印字体
                301: "/usr/share/fonts/truetype/moon_get-Heavy.ttf",  # 日文 Gothic
                # 日文优先字体
                100: "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",  # 日文 Gothic
                101: "/usr/share/fonts/truetype/fonts-japanese-mincho.ttf",  # 日文 Mincho
                102: "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto CJK
                # 系统字体
                1: "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",      # Noto Sans - 通用字体
                2: "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",   # Noto Sans CJK - 中日韩
                3: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",         # DejaVu Sans
                4: "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Liberation Sans
                5: "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",           # Ubuntu Regular
                6: "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", # Droid Sans
                7: "/usr/share/fonts/truetype/freefont/FreeSans.ttf",         # Free Sans
                8: "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",  # Liberation Mono
                9: "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",    # Noto Serif
                10: "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",    # Noto Sans CJK Bold
                # 通用备用字体
                200: "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                201: "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
            }
            
            # 加载字体
            font = None
            
            # 1. 尝试加载指定字体
            if subtitle_font is not None and subtitle_font in subtitle_font_paths:
                font_path = subtitle_font_paths[subtitle_font]
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, fontsize)
                        self.logger.info(f"Successfully loaded specified font: {os.path.basename(font_path)}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load specified font: {str(e)}")
            
            # 2. 如果指定字体加载失败，根据语言选择合适的字体
            if font is None:
                font_priority = [100, 101, 102] if is_jp else [200, 201]  # 根据语言选择优先字体
                for font_id in font_priority:
                    font_path = subtitle_font_paths.get(font_id)
                    if font_path and os.path.exists(font_path):
                        try:
                            font = ImageFont.truetype(font_path, fontsize)
                            self.logger.info(f"Using font: {os.path.basename(font_path)}")
                            break
                        except Exception:
                            continue
            
            # 3. 如果仍然失败，使用默认字体
            if font is None:
                self.logger.warning("Using default font as fallback")
                font = ImageFont.load_default()
            
            # 测试字体是否能正确渲染文本
            try:
                # 获取文本大小
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                self.logger.info(f"Text size: {text_width}x{text_height}")
            except Exception as e:
                self.logger.error(f"Error measuring text: {str(e)}")
                return frame
            
            # 计算文本位置
            x, y = self._calculate_text_position(position, frame_size, text_width, text_height)
            
            # 添加文本描边（增强可读性）
            outline_color = (0, 0, 0, 255)
            outline_width = 2
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                for i in range(outline_width):
                    draw.text((x + dx + i, y + dy + i), text, font=font, fill=outline_color)
            
            # 添加主文本
            draw.text((x, y), text, font=font, fill=(*color, 255))
            
            return np.array(pil_img)
            
        except Exception as e:
            self.logger.error(f"Error creating text frame: {str(e)}", exc_info=True)
            return np.zeros((frame_size[1], frame_size[0], 4), dtype=np.uint8)
        
    


    '''
    升级版本
    '''
    def merge_clips(self, clip_mapping, clips, merged_video_path):
        if not clip_mapping:
            raise ValueError("No clips were successfully downloaded")
        
        try:
            return self.concatenate_videos(clip_mapping, clips, merged_video_path)
        except Exception as e:
            self.logger.error(f"Error in merge_clips: {str(e)}", exc_info=True)
            raise

    


    


    def apply_effects(self, video_path, preset,narration,project_no):
        try:
            # 加载原始视频
            video = VideoFileClip(video_path)
            frame_size = video.size
            final_video = video
            narration_audio_path = None


            # 处理旁白
            if preset.get('narration') and narration != "":
                self.logger.info('=======> start processing narration')
                narration_content = narration
                if narration_content is not None:
       
                    voice_map = self.llm_util.get_voice_id()
                    self.logger.info(f'voice map is: {voice_map}')
                    narration_voice = preset.get('narrationVoice', 0)
                    narration_style = preset.get('narrationStyle', 0)
                    
                    # 根据性别和风格选择声音
                    voice_id = self._select_voice_id(voice_map, narration_voice, narration_style)
                    
                    if voice_id:
                        '''
                        使用elevenlab音频
                        '''
                        self.logger.info(f'selected voice id is: {voice_id}')
                        narration_audio_path = self.llm_util.text_to_speech_by_elevenlabs(
                            narration_content, 
                            voice_id=voice_id
                        )
                    else:
                        self.logger.warning("No suitable voice found, using default voice")
                        narration_audio_path = self.llm_util.text_to_speech(narration_content)

                    # narration_audio_path = self.llm_util.text_to_speech(narration_content)
                    self.logger.info(f'Narration audio path: {narration_audio_path}')
                
                    if narration_audio_path and os.path.exists(narration_audio_path):
                        narration_audio = AudioFileClip(narration_audio_path)
                        if final_video.audio is None:
                            final_video = final_video.set_audio(narration_audio)
                        else:
                            final_video = final_video.set_audio(CompositeAudioClip([final_video.audio, narration_audio]))


            # 如果没有旁白，使用原始视频的音频内容作为字幕
            if not preset.get('narration') and preset.get('subtitle'):
                self.logger.info('=======> Adding subtitles from original video content')
                try:
                    # 提取原始视频的音频用于生成字幕
                    audio_path = self.extract_audio(project_no, video_path)
                    if not audio_path:
                        self.logger.warning("Failed to extract audio from original video")
                        return final_video

                    # 获取音频转写结果
                    transcription_result = self.perform_asr(audio_path)
                    if transcription_result and transcription_result.get('transcription'):
                        # 只设置字幕文本，不处理音频
                        narration = ' '.join([segment['text'] for segment in transcription_result['transcription']])
                        narration_audio_path = audio_path
                    
                except Exception as e:
                    self.logger.error(f"Error processing original content for subtitles: {str(e)}", exc_info=True)
            

            
            '''
            fc-list : family
            sudo apt-get update
            sudo apt-get install -y imagemagick
            sudo apt-get remove --purge imagemagick
            sudo apt-get autoremove
            sudo apt-get update
            sudo apt-get install imagemagick

            sudo nano /etc/ImageMagick-6/policy.xml
            <policymap>
            <!-- ... other policies ... -->
            <policy domain="path" rights="read,write" pattern="@*" />
            <policy domain="coder" rights="read,write" pattern="*" />
            <!-- 如果有这行，请注释掉或删除：
            <policy domain="path" rights="none" pattern="@*" />
            -->
            </policymap>

            convert -version
            '''
            if preset.get('subtitle'):
                try:
                    self.logger.info('======> start processing subtitle and narration')
                    
                    if not narration or not narration_audio_path:
                        self.logger.warning("Missing narration text or audio path for subtitles")
                        return final_video
                    
                    # 1. 加载旁白音频以获取总时长
                    narration_audio = AudioFileClip(narration_audio_path)
                    total_duration = narration_audio.duration
                    
                    # 2. 将旁白文本分割成句子
                    sentences = re.split('[。！？.!?]', narration)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    
                    # 3. 根据句子长度分配时间
                    total_chars = sum(len(s) for s in sentences)
                    time_per_char = total_duration / total_chars
                    
                    # 4. 计算每个句子的时间戳
                    subtitle_segments = []
                    current_time = 0
                    
                    for sentence in sentences:
                        if not sentence:
                            continue
                        duration = len(sentence) * time_per_char
                        subtitle_segments.append({
                            'text': sentence,
                            'start': current_time,
                            'end': current_time + duration
                        })
                        current_time += duration
                    
                    def split_text_into_lines(text, max_chars_per_line):
                        """将文本分割成多行"""
                        if len(text) <= max_chars_per_line:
                            return [text]
                            
                        lines = []
                        current_line = ""
                        
                        # 按标点符号和空格分割
                        parts = re.findall(r'[^，。！？,.!?\s]+[，。！？,.!?\s]?', text)
                        
                        for part in parts:
                            if len(current_line + part) <= max_chars_per_line:
                                current_line += part
                            else:
                                if current_line:
                                    lines.append(current_line)
                                current_line = part
                                
                                # 如果单个部分超过最大长度，强制分割
                                while len(current_line) > max_chars_per_line:
                                    lines.append(current_line[:max_chars_per_line])
                                    current_line = current_line[max_chars_per_line:]
                                    
                        if current_line:
                            lines.append(current_line)
                            
                        return lines
                    

                    # 5. 创建字幕处理函数
                    def process_frame(get_frame, t):
                        # 获取当前帧
                        frame = get_frame(t)
                        frame_width = frame.shape[1]
                        
                        # 检查是否需要添加字幕
                        if preset.get('subtitleFont', 0) == 0 or preset.get('subtitleStyle', 0) == 0:
                            return frame
                        
                        # 找到当前时间应该显示的句子
                        current_subtitle = None
                        for segment in subtitle_segments:
                            if segment['start'] <= t <= segment['end']:
                                current_subtitle = segment['text']
                                break
                        
                        if not current_subtitle:
                            return frame
                        
                        # 设置字幕样式
                        fontsize = {
                            0: 0,     # 不显示字幕
                            1: 24,    # 小
                            2: 48,    # 中
                            3: 72     # 大
                        }.get(preset.get('subtitleStyle', 2), 48)  # 默认使用中等大小
                        
                        # 如果字体大小为0，不显示字幕
                        if fontsize == 0:
                            return frame
                        
                         # 计算每行最大字符数（根据视频宽度和字体大小估算）
                        max_chars_per_line = int(frame_width / (fontsize * 0.7))  # 0.7是一个经验系数
                        
                        # 将字幕文本分割成多行
                        subtitle_lines = split_text_into_lines(current_subtitle, max_chars_per_line)
                        
                        # 将多行文本合并，用换行符连接
                        formatted_subtitle = '\n'.join(subtitle_lines)

                        # 将16进制颜色转换为RGB
                        try:
                            hex_color = preset.get('subtitleColor', '#FFFFFF').lstrip('#')
                            font_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        except Exception as e:
                            self.logger.warning(f"Error parsing subtitle color: {str(e)}, using default white")
                            font_color = (255, 255, 255)
                        
                         # 创建当前字幕的帧
                        subtitle_frame = self.create_text_frame(
                            formatted_subtitle,
                            frame.shape[:2][::-1],
                            fontsize=fontsize,
                            color=font_color,
                            position='bottom',
                            subtitle_font=preset.get('subtitleFont', None)
                        )

                        # 转换帧为RGBA
                        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                        
                        # 计算字幕的 alpha 遮罩
                        alpha_subtitle = subtitle_frame[:, :, 3:] / 255.0
                        alpha_frame = 1.0 - alpha_subtitle
                        
                        # 添加半透明黑色背景
                        bg_color = np.zeros_like(frame_rgba)
                        bg_color[:, :, 3] = alpha_subtitle[:, :, 0] * 128
                        
                        # 混合原始帧、背景和字幕
                        for c in range(3):
                            frame_rgba[:, :, c] = (
                                frame_rgba[:, :, c] * alpha_frame[:, :, 0] +
                                bg_color[:, :, c] * (alpha_subtitle[:, :, 0] * 0.5) +
                                subtitle_frame[:, :, c] * alpha_subtitle[:, :, 0]
                            )
                        
                        return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)
                    
                    # 6. 应用字幕到视频
                    self.logger.info('Processing frames with timed subtitles...')
                    final_video = final_video.fl(process_frame)
                    
                except Exception as e:
                    self.logger.error(f"Error processing frames with timed subtitles: {str(e)}", exc_info=True)



            # 处理音频和字幕
            if preset.get('narration') and narration_audio_path:
                # 有旁白时，混合音频并添加字幕
                self.logger.info("Adding narration audio and subtitles...")
                if final_video.audio is None:
                    final_video = final_video.set_audio(narration_audio)
                else:
                    # 降低原始音频音量到30%
                    original_audio = final_video.audio.volumex(0.3)
                    final_video = final_video.set_audio(
                        CompositeAudioClip([original_audio, narration_audio])
                    )
                self.logger.info("Successfully mixed narration with lowered original audio")
            else:
                # 没有旁白时，只添加字幕，保持原始音频不变
                self.logger.info("Keeping original audio and adding subtitles only...")
                # 不对音频做任何处理，保持原样
                

            # 调整视频时长
            final_video = self.adjust_video_duration(final_video, narration_audio_path)


            # 添加水印
            if preset.get('watermark'):
                self.logger.info('=======> start processing watermark')
                watermark_text = preset.get('watermarkText', '')
                
                
                    # 水印位置映射
                position_mapping = {
                        0: 'bottom',      # 默认底部居中
                        1: 'top-left',    # 左上
                        2: 'top-right',   # 右上
                        3: 'center',      # 中间
                        4: 'bottom-left', # 左下
                        5: 'bottom-right' # 右下
                }

                # 获取位置设置，默认为底部居中
                position_value = position_mapping.get(preset.get('watermarkPosition', 0), 'bottom')
                    


                watermark_frame = self.create_text_frame(
                    watermark_text, 
                    frame_size,
                    fontsize=72,
                    position= position_value,
                    subtitle_font=301 #水印字体
                )
                


                def add_watermark_to_frame(frame):
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    
                    # 设置水印不透明度为30%（透明度70%）
                    alpha_watermark = (watermark_frame[:, :, 3:] / 255.0) * 0.3  # Changed to 0.3 for 30% opacity
                    alpha_frame = 1.0 - alpha_watermark
                    
                    # 简化水印叠加过程，移除多重偏移效果
                    result = frame_rgba.copy()
                    for c in range(3):
                        result[:, :, c] = (result[:, :, c] * alpha_frame[:, :, 0] + 
                                         watermark_frame[:, :, c] * alpha_watermark[:, :, 0])
                    
                    return cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)
                


                print('Processing frames with watermark...')
                final_video = final_video.fl_image(add_watermark_to_frame)
                self.logger.info("Added watermark to all frames")


            # 处理背景音乐
            if preset.get('bgm') and preset.get('bgm') != '':  # Check if BGM is specified and not empty
                self.logger.info('=======> Starting BGM processing')
                bgm_type = preset.get('bgm')
                bgm = None
                
                try:
                    # Get BGM path
                    bgm_path = self._get_bgm_path(bgm_type)
                    self.logger.info(f'Selected BGM path: {bgm_path}')
                    
                    # Validate BGM file exists
                    if not os.path.exists(bgm_path):
                        self.logger.warning(f"BGM file not found at path: {bgm_path}")
                        return final_video
                        
                    # Load and process BGM
                    bgm = AudioFileClip(bgm_path)
                    
                    # Match BGM duration to video duration
                    if bgm.duration < final_video.duration:
                        self.logger.info("BGM shorter than video - looping BGM")
                        bgm = bgm.loop(duration=final_video.duration)
                    elif bgm.duration > final_video.duration:
                        self.logger.info("BGM longer than video - trimming BGM")
                        bgm = bgm.subclip(0, final_video.duration)
                    
                    # Adjust BGM volume (optional)
                    bgm = bgm.volumex(0.3)  # Reduce BGM volume to 30%
                    
                    # Combine with existing audio
                    if final_video.audio is None:
                        self.logger.info("No existing audio - setting BGM as main audio")
                        final_video = final_video.set_audio(bgm)
                    else:
                        self.logger.info("Combining existing audio with BGM")
                        final_video = final_video.set_audio(
                            CompositeAudioClip([final_video.audio, bgm])
                        )
                        
                    self.logger.info("Successfully added BGM to video")
                    
                except Exception as e:
                    self.logger.error(f"Error processing BGM: {str(e)}", exc_info=True)



            # 保存最终视频
            self.logger.info('=====> Saving final video...')
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_path = temp_output.name
            temp_output.close()


            final_video.write_videofile(
                temp_path,
                codec="libx264",
                audio_codec="aac",
                fps=video.fps,
                threads=4
            )

            # 移动到最终位置
            shutil.move(temp_path, video_path)
            return video_path

        except Exception as e:
            self.logger.error(f"Error applying effects to video: {str(e)}", exc_info=True)
            return None
        finally:
            if 'video' in locals():
                video.close()
            if 'final_video' in locals() and final_video != video:
                final_video.close()

    
    def adjust_video_duration(self, video_clip, narration_audio_path, max_diff_seconds=5):
        """
        调整视频时长以匹配旁白音频
        
        Args:
            video_clip (VideoFileClip): 需要调整的视频片段
            narration_audio_path (str): 旁白音频文件路径
            max_diff_seconds (int): 允许的最大时长差异（秒）
            
        Returns:
            VideoFileClip: 调整后的视频片段
        """
        if not narration_audio_path or not os.path.exists(narration_audio_path):
            self.logger.warning("No narration audio file found for duration adjustment")
            return video_clip

        try:
            self.logger.info('=====> Adjusting video duration to match narration...')
            narration_audio = AudioFileClip(narration_audio_path)
            
            try:
                narration_duration = narration_audio.duration
                video_duration = video_clip.duration
                
                # 计算时长差异
                duration_diff = abs(video_duration - narration_duration)
                
                if duration_diff > max_diff_seconds:
                    self.logger.info(f'Video duration ({video_duration:.2f}s) differs from narration ({narration_duration:.2f}s) by {duration_diff:.2f}s')
                    
                    # 选择较短的时长，并给予0.5秒的缓冲
                    target_duration = min(video_duration, narration_duration) + 0.5
                    
                    # 裁剪视频
                    adjusted_video = video_clip.subclip(0, target_duration)
                    self.logger.info(f'Adjusted video duration to {target_duration:.2f}s')
                    return adjusted_video
                else:
                    self.logger.info(f'Video and narration duration difference ({duration_diff:.2f}s) within acceptable range')
                    return video_clip
                    
            finally:
                narration_audio.close()
                
        except Exception as e:
            self.logger.error(f"Error adjusting video duration: {str(e)}", exc_info=True)
            return video_clip


    def perform_asr(self, audio_path):
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            audio = AudioSegment.from_wav(audio_path)
            chunk_size = 20 * 1024 * 1024  # 20MB in bytes
            chunk_length_ms = (chunk_size / len(audio.raw_data)) * len(audio)
            timestamped_transcription = []
            total_chunks = math.ceil(len(audio) / chunk_length_ms)
            
            for i, chunk_start in tqdm(enumerate(range(0, len(audio), int(chunk_length_ms))), total=total_chunks, desc="Processing audio chunks"):
                try:
                    chunk_end = chunk_start + chunk_length_ms
                    chunk = audio[chunk_start:chunk_end]
                    
                    chunk_path = f"{audio_path}_chunk_{i}.wav"
                    chunk.export(chunk_path, format="wav")
                    
                    chunk_transcription = self.llm_util.audio_to_srt(chunk_path)
                    self.logger.info(f'Chunk {i} transcription: {chunk_transcription[:100]}...')  # Log first 100 chars
                    segments = self._parse_transcript(chunk_transcription, chunk_start)
                    timestamped_transcription.extend(segments)
                    
                except Exception as e:
                    self.logger.error(f"Error processing chunk {i}: {str(e)}")
                finally:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
            
            # Save ASR results to JSON file
            asr_result = {
                "audio_file": os.path.basename(audio_path),
                "transcription": timestamped_transcription
            }
            
            return asr_result

        except FileNotFoundError as e:
            self.logger.error(f"File not found error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"An error occurred during ASR: {str(e)}")
            raise
    
    '''
    开始用LLM进行分析
    '''
    def select_relevant_transcription(self, transcription, clip_desc, preset):
        """
        Select relevant transcription segments based on clip description and language
        
        Args:
            transcription (dict): Original transcription data
            clip_desc (str): Clip description
            preset (dict): Preset settings including language
            
        Returns:
            dict: Selected transcription segments in the required format
            None: If processing fails
        """
        try:
            lang = preset.get('narrationLang', 'zh')
            
            # 根据语言设置提示语
            prompts = {
                'zh': {
                    'main': '''根据剪辑描述，从以下待剪辑片段中选择符合要求的音频文本。返回选中的片段，保持原始格式不变,按照要求的JSON格式返回。

                    待剪辑片段:
                    {transcription}

                    剪辑描述: {clip_desc}

                    约束：
                    - 返回格式必须与输入的"待剪辑片段"格式完全一致
                    - 只返回你认为符合剪辑要求的片段
                    - 可以选择一个或多个文件中的片段
                    - 每个文件可以选择一个或多个片段
                    - 所有文本必须是中文
                    - 不要添加任何额外解释或修改原始数据结构
                    ''',
                    'error': '输出必须是中文'
                },
                'en': {
                    'main': '''Based on the clip description, select relevant audio text segments from the following clips. Return the selected segments in the original format as required JSON.

                    Clips to process:
                    {transcription}

                    Clip description: {clip_desc}

                    Constraints:
                    - Return format must match the input format exactly
                    - Only return segments that match the requirements
                    - Can select segments from one or multiple files
                    - Can select one or multiple segments per file
                    - All text must be in English
                    - Do not add any extra explanations or modify the original data structure
                    ''',
                    'error': 'Output must be in English'
                },
                'ja': {
                    'main': '''クリップの説明に基づいて、以下のクリップから関連する音声テキストセグメントを選択してください。選択したセグメントを元の形式で必要なJSONとして返してください。

                    処理するクリップ:
                    {transcription}

                    クリップの説明: {clip_desc}

                    制約：
                    - 返却フォーマットは入力フォーマットと完全に一致する必要があります
                    - 要件に一致するセグメントのみを返してください
                    - 1つまたは複数のファイルからセグメントを選択できます
                    - ファイルごとに1つまたは複数のセグメントを選択できます
                    - すべてのテキストは日本語である必要があります
                    - 追加の説明を加えたり、元のデータ構造を変更したりしないでください
                    ''',
                    'error': '出力は日本語である必要があります'
                }
            }
            
            # 获取对应语言的提示语
            prompt_template = prompts.get(lang, prompts['zh'])
            
            # 格式化提示语
            user_prompt = prompt_template['main'].format(
                transcription=json.dumps(transcription, ensure_ascii=False, indent=2),
                clip_desc=clip_desc
            )
            
            try:
                # 调用native OpenAI chat接口
                response = self.llm_util.native_structured_chat(
                    message=user_prompt,
                    object_define=TranscriptionResponse
                )
                
                if response and hasattr(response, 'parsed'):
                    # 将Pydantic模型转换为字典
                    result = response.parsed.model_dump()
                        
                    self.logger.info(f"====> Selected transcription segments: {result}")
                    return result
                else:
                    self.logger.error("Invalid response format received")
                    return None
                
            except Exception as e:
                self.logger.error(f"Error in selecting transcription segments: {str(e)}", exc_info=True)
                return None
        
        except Exception as e:
            self.logger.error(f"Error in transcription selection: {str(e)}", exc_info=True)
            return None


    
    def generate_title(self, transcription, preset):
        """
        根据视频内容生成标题，支持多语言
        
        Args:
            transcription (dict): 视频文本内容
            preset (dict): 预设配置，包含语言设置
            
        Returns:
            str: 生成的标题
            None: 如果生成失败
        """
        try:
            lang = preset.get('narrationLang', 'zh')
            
            # 多语言提示模板
            prompts = {
                'zh': {
                    'main': '''根据以下视频片段的文本内容，生成一个简洁的标题。

                    视频片段:
                    {transcription}

                    约束：
                    - 标题必须是中文
                    - 标题应简洁明了，不超过20个字
                    - 能够准确反映视频内容的核心主题
                    - 使用自然语言，避免使用过于专业或晦涩的词汇
                    - 不要添加任何额外解释
                    - 直接返回标题文本
                    ''',
                    'error': '标题必须是中文'
                },
                'en': {
                    'main': '''Generate a concise title based on the following video content.

                    Video content:
                    {transcription}

                    Constraints:
                    - Title must be in English
                    - Keep it concise, no more than 10 words
                    - Accurately reflect the core theme of the video
                    - Use natural language, avoid technical jargon
                    - No additional explanations
                    - Return only the title text
                    ''',
                    'error': 'Title must be in English'
                },
                'ja': {
                    'main': '''以下の動画コンテンツに基づいて、簡潔なタイトルを生成してください。

                    動画コンテンツ:
                    {transcription}

                    制約：
                    - タイトルは日本語で書かれている必要があります
                    - 簡潔で、20文字以内にしてください
                    - 動画の核心的なテーマを正確に反映すること
                    - 自然な言葉を使用し、専門用語や難解な表現を避けること
                    - 追加の説明を付けないこと
                    - タイトルのテキストのみを返すこと
                    ''',
                    'error': 'タイトルは日本語である必要があります'
                }
            }
            
            # 获取对应语言的提示语
            prompt_template = prompts.get(lang, prompts['zh'])
            
            # 格式化提示语
            user_prompt = prompt_template['main'].format(
                transcription=json.dumps(transcription, ensure_ascii=False, indent=2)
            )
            
            # 调用OpenAI chat接口
            title = self.llm_util.native_chat(user_prompt)
                
            # 清理标题文本（去除引号、空格等）
            title = self._clean_title(title, lang)
            
            self.logger.info(f"Generated title: {title}")
            return title
            
        except Exception as e:
            self.logger.error(f"Error generating title: {str(e)}")
            return None
            

        
    def generate_description(self, transcription, preset):
        """
        根据视频内容生成描述，支持多语言
        
        Args:
            transcription (dict): 视频文本内容
            preset (dict): 预设配置，包含语言设置
            
        Returns:
            str: 生成的描述
            None: 如果生成失败
        """
        try:
            lang = preset.get('narrationLang', 'zh')
            
            # 多语言提示模板
            prompts = {
                'zh': {
                    'main': '''根据以下视频片段的文本内容，生成一个简洁的描述。

                    视频片段:
                    {transcription}

                    约束：
                    - 描述必须是中文
                    - 描述长度应在50-100字之间
                    - 内容要完整且有逻辑性
                    - 使用自然流畅的语言
                    - 避免使用过于专业或晦涩的词汇
                    - 分段落组织，每段不超过30字
                    - 不要添加任何额外解释
                    - 直接返回描述文本
                    ''',
                    'error': '描述必须是中文'
                },
                'en': {
                    'main': '''Generate a concise description based on the following video content.

                    Video content:
                    {transcription}

                    Constraints:
                    - Description must be in English
                    - Length should be between 30-50 words
                    - Content should be complete and logical
                    - Use natural and fluent language
                    - Avoid technical jargon
                    - Organize in paragraphs, each no more than 20 words
                    - No additional explanations
                    - Return only the description text
                    ''',
                    'error': 'Description must be in English'
                },
                'ja': {
                    'main': '''以下の動画コンテンツに基づいて、簡潔な説明文を生成してください。

                    動画コンテンツ:
                    {transcription}

                    制約：
                    - 説明文は日本語で書かれている必要があります
                    - 長さは100-200文字程度にしてください
                    - 内容は完全で論理的である必要があります
                    - 自然で流暢な言葉を使用してください
                    - 専門用語や難解な表現を避けてください
                    - 段落で構成し、各段落は50文字以内にしてください
                    - 追加の説明を付けないでください
                    - 説明文のテキストのみを返してください
                    ''',
                    'error': '説明文は日本語である必要があります'
                }
            }
            
            # 获取对应语言的提示语
            prompt_template = prompts.get(lang, prompts['zh'])
            
            # 格式化提示语
            user_prompt = prompt_template['main'].format(
                transcription=json.dumps(transcription, ensure_ascii=False, indent=2)
            )
            
            # 调用OpenAI chat接口
            description = self.llm_util.native_chat(user_prompt)
                
            # 清理和格式化描述文本
            description = self._format_description(description, lang)
            
            self.logger.info(f"Generated description: {description}")
            return description
            
        except Exception as e:
            self.logger.error(f"Error generating description: {str(e)}")
            return None
            
    

    def generate_narration(self, transcription, preset):
        """
        根据视频内容生成旁白，支持多语言
        
        Args:
            transcription (dict): 视频文本内容
            preset (dict): 预设配置，包含语言和旁白风格设置
            
        Returns:
            str: 生成的旁白
            None: 如果生成失败
        """
        try:
            lang = preset.get('narrationLang', 'zh')
            narration_style = preset.get('narrationStyle', 0)
            
            # 根据旁白风格选择语气
            style_tone = {
                0: '',  # 默认
                1: '慈祥温和的语气',  # 慈祥
                2: '文静优雅的语气',  # 文静
                3: '轻松幽默的语气',  # 幽默
                4: '雄壮有力的语气',  # 雄壮
                5: '和蔼可亲的语气'   # 和蔼
            }.get(narration_style, '')
            
            # 多语言提示模板
            prompts = {
                'zh': {
                    'main': '''根据以下视频片段的文本内容，生成一个简洁的旁白。

                    视频片段:
                    {transcription}

                    约束：
                    - 旁白必须是中文
                    - 使用{style}
                    - 旁白应简洁自然，每句话不超过30个字
                    - 语言要生动形象，适合口语表达
                    - 避免使用过于专业或晦涩的词汇
                    - 注意语气的连贯性和流畅度
                    - 直接返回旁白文本，不要加引号
                    - 不要添加任何额外解释
                    ''',
                    'error': '旁白必须是中文'
                },
                'en': {
                    'main': '''Generate a concise narration based on the following video content.

                    Video content:
                    {transcription}

                    Constraints:
                    - Narration must be in English
                    - Use {style}
                    - Keep each sentence concise, no more than 15 words
                    - Use vivid and natural language suitable for speaking
                    - Avoid technical jargon
                    - Maintain consistent tone and flow
                    - Return only the narration text, no quotation marks
                    - No additional explanations
                    ''',
                    'error': 'Narration must be in English'
                },
                'ja': {
                    'main': '''以下の動画コンテンツに基づいて、簡潔なナレーションを生成してください。

                    動画コンテンツ:
                    {transcription}

                    制約：
                    - ナレーションは日本語で書かれている必要があります
                    - {style}を使用してください
                    - 各文は簡潔で、30文字以内にしてください
                    - 話し言葉に適した生き生きとした自然な言葉を使用してください
                    - 専門用語や難解な表現を避けてください
                    - 一貫した口調とスムーズな流れを維持してください
                    - ナレーションテキストのみを返し、引用符は付けないでください
                    - 追加の説明を付けないでください
                    ''',
                    'error': 'ナレーションは日本語である必要があります'
                }
            }
            
            # 获取对应语言的提示语
            prompt_template = prompts.get(lang, prompts['zh'])
            
            # 根据语言调整风格描述
            style_translations = {
                'zh': {
                    1: '慈祥温和的语气',
                    2: '文静优雅的语气',
                    3: '轻松幽默的语气',
                    4: '雄壮有力的语气',
                    5: '和蔼可亲的语气'
                },
                'en': {
                    1: 'a warm and gentle tone',
                    2: 'a calm and elegant tone',
                    3: 'a light and humorous tone',
                    4: 'a strong and powerful tone',
                    5: 'a kind and friendly tone'
                },
                'ja': {
                    1: '温かく優しい口調',
                    2: '落ち着いた上品な口調',
                    3: '軽やかでユーモアのある口調',
                    4: '力強い口調',
                    5: '親しみやすい優しい口調'
                }
            }
            
            style = style_translations.get(lang, {}).get(narration_style, '')
            
            # 格式化提示语
            user_prompt = prompt_template['main'].format(
                transcription=json.dumps(transcription, ensure_ascii=False, indent=2),
                style=style
            )
            
            # 调用OpenAI chat接口
            narration = self.llm_util.native_chat(user_prompt)
            
            # 清理旁白文本
            narration = self._clean_narration(narration, lang)
            
            self.logger.info(f"Generated narration: {narration}")
            return narration
            
        except Exception as e:
            self.logger.error(f"Error generating narration: {str(e)}")
            return None
            



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



    def concatenate_videos(self, clip_mapping,clips, merged_video_path):
        if not clip_mapping:
            raise ValueError("No clips were successfully downloaded")
    
    
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

            for index, clip in enumerate(clips):
                self.logger.info(f"Processing clip {index}: {clip}")
                if clip['type'] == 'video':
                    video_url = clip.get('videoUrl')
                    if not video_url:
                        self.logger.error(f"Invalid video URL for clip {index}: {clip}")
                        continue
                    
                    video_local_path = clip_mapping.get(video_url)
                    if not video_local_path:
                        self.logger.error(f"No local path found for video URL: {video_url}")
                        continue
                    
                    if not os.path.exists(video_local_path):
                        self.logger.error(f"Local video file not found: {video_local_path}")
                        continue
                    
                    try:
                        video_clip = VideoFileClip(video_local_path)
                    except Exception as e:
                        self.logger.error(f"Error loading video clip {index} from path {video_local_path}: {str(e)}")
                        continue
                    
                    # 应用转场效果
                    transition_effect = clip.get('transitionEffect', 0)
                    if transition_effect == 1:
                        video_clip = video_clip.fx(vfx.fadein, duration=0.5)  # Changed from fade_in to fadein
                    elif transition_effect == 2:
                        video_clip = video_clip.fx(vfx.fadeout, duration=0.5)  # Changed from fade_out to fadeout
                    
                    video_clips.append(video_clip)

                elif clip['type'] == 'effect':
                    # 处理效果类型的片段
                    transition_effect = clip.get('transitionEffect', 0)
                    if transition_effect == 1:
                        # 添加淡入效果到下一个视频片段
                        if video_clips:
                            video_clips[-1] = video_clips[-1].fx(vfx.fadeout, duration=0.5)  # Changed from fade_out to fadeout
                    elif transition_effect == 2:
                        # 添加淡出效果到上一个视频片段
                        if video_clips:
                            video_clips[-1] = video_clips[-1].fx(vfx.fadeout, duration=0.5)  # Changed from fade_out to fadeout
                else:
                    self.logger.warning(f"Unknown clip type for clip {index}: {clip['type']}")

            if not video_clips:
                raise ValueError("No valid video clips to concatenate")

            # 合并视频片段
            self.logger.info(f"Concatenating {len(video_clips)} video clips")
            final_clip = concatenate_videoclips(video_clips, method="compose")

            # 写入临时文件
            self.logger.info(f"Writing concatenated video to temporary file: {temp_output_path}")
            final_clip.write_videofile(temp_output_path, codec="libx264", audio_codec="aac")

            # 如果最终输出文件已存在，则删除
            if os.path.exists(merged_video_path):
                os.remove(merged_video_path)

            # 复制临时文件到最终位置，然后删除临时文件
            shutil.copy2(temp_output_path, merged_video_path)

            self.logger.info(f"Video successfully merged and saved to: {merged_video_path}")
            return merged_video_path

        except Exception as e:
            self.logger.error(f"Error during video concatenation: {str(e)}", exc_info=True)
            return None

        finally:
            # 清理资源
            for clip in video_clips:
                clip.close()
            if 'final_clip' in locals():
                final_clip.close()
            if temp_output_path and os.path.exists(temp_output_path):
                os.remove(temp_output_path)

        
    
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
    


    def post_process_clips(self,transcription,preset):
        try:
            post_result = {
                "title": "",
                "description": "",
                "narration": ""
            }

            # 生成标题
            post_result['title'] = self.generate_title(transcription,preset)

            # 生成描述
            post_result['description'] = self.generate_description(transcription,preset)

            # 生成旁白
            post_result['narration'] = self.generate_narration(transcription,preset)


            return post_result
        
        except Exception as e:
            print(f"Error in post_process_video: {str(e)}")
            raise
        

    def _parse_transcript(self, transcript, chunk_start_ms):
        lines = transcript.split('\n')
        segments = []
        current_segment = None

        for line in lines:
            if re.match(r'\d+$', line):  # 序号行，忽略
                continue
            elif '-->' in line:  # 时间戳行
                if current_segment:
                    segments.append(current_segment)
                start, end = self._parse_timestamp(line, chunk_start_ms)
                current_segment = {'start': start, 'end': end, 'text': ''}
            elif line.strip():  # 文本行
                if current_segment:
                    current_segment['text'] += line.strip() + ' '

        if current_segment:  # 添加最后一个段落
            segments.append(current_segment)

        return segments

    def _parse_timestamp(self, timestamp_line, chunk_start_ms):
        start, end = timestamp_line.split('-->')
        start = self._time_to_ms(start.strip()) + chunk_start_ms
        end = self._time_to_ms(end.strip()) + chunk_start_ms
        return start / 1000, end / 1000  # 转换为秒

    def _time_to_ms(self, time_str):
        h, m, s = time_str.split(':')
        s, ms = s.split(',')
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
    
    def _get_watermark_position(self, position):
        positions = {
            1: ('left', 'top'),
            2: ('right', 'top'),
            3: ('left', 'bottom'),
            4: ('right', 'bottom'),
            5: ('center', 'center')
        }
        return positions.get(position, ('right', 'bottom'))
    
    '''
    0=无
    1=欢快
    2=旅途
    3=忧伤
    4=安静
    5=热烈
    '''
    def _get_bgm_path(self, bgm_type):
        # Define a mapping of bgm types to file paths
        bgm_paths = {
            1: os.path.join(os.getcwd(), 'music', 'happy.MP3'),
            2: os.path.join(os.getcwd(), 'music', 'journey.MP3'),
            3: os.path.join(os.getcwd(), 'music', 'sad.MP3'),
            4: os.path.join(os.getcwd(), 'music', 'quiet.MP3'),
            5: os.path.join(os.getcwd(), 'music', 'warm.MP3')
        }
        return bgm_paths.get(bgm_type, os.path.join(os.getcwd(),'music','quiet.mp3'))
    

    def _clean_title(self, title, lang):
        """
        清理标题文本
        """
        # 去除可能的引号和多余空格
        title = title.strip(' \'"')
        
        # 根据语言进行特定处理
        if lang == 'zh':
            # 移除中文标点
            title = re.sub(r'[，。！？、：；]', '', title)
        elif lang == 'en':
            # 确保英文标题大小写正确
            title = title.title()
            # 移除英文标点
            title = re.sub(r'[,.!?:;]', '', title)
        elif lang == 'ja':
            # 移除日文标点
            title = re.sub(r'[、。！？：；]', '', title)
            
        return title.strip()


    def _calculate_text_position(self, position, frame_size, text_width, text_height):
        """
        计算文本位置
        
        Args:
            position (str): 位置标识
            frame_size (tuple): 帧大小 (width, height)
            text_width (int): 文本宽度
            text_height (int): 文本高度
            
        Returns:
            tuple: (x, y) 坐标
        """
        padding = 20  # 边距
        
        # 位置映射字典
        positions = {
            'diagonal-center': (
                (frame_size[0] - text_width) // 2 + frame_size[0] // 8,
                (frame_size[1] - text_height) // 2 + frame_size[1] // 8
            ),
            'top-left': (
                padding,
                padding + text_height
            ),
            'top-right': (
                frame_size[0] - text_width - padding,
                padding + text_height
            ),
            'center': (
                (frame_size[0] - text_width) // 2,
                (frame_size[1] - text_height) // 2
            ),
            'bottom-left': (
                padding,
                frame_size[1] - text_height - padding
            ),
            'bottom-right': (
                frame_size[0] - text_width - padding,
                frame_size[1] - text_height - padding
            ),
            'bottom': (
                (frame_size[0] - text_width) // 2,
                frame_size[1] - text_height - padding * 4  # 增加了padding的倍数，使文字位置更靠上
            ),
            'top': (
                (frame_size[0] - text_width) // 2,
                padding + text_height
            )
        }
        
        # 获取位置，默认为底部居中
        return positions.get(position, positions['bottom'])
    
    

    def _extract_tags(self, tags_text):
        """从文本中提取标签列表"""
        try:
            # 移除常见的标签前缀
            clean_text = tags_text.replace('标签：', '').replace('Tags:', '').replace('タグ：', '')
            
            # 分割文本获取标签
            tags = []
            for tag in re.split(r'[,，、\n]', clean_text):
                tag = tag.strip().strip('#').strip()
                if tag and len(tag) > 0:
                    tags.append(tag)
            
            return list(set(tags))  # 去重
        except Exception as e:
            self.logger.error(f"Error extracting tags: {str(e)}")
            return []

    def _calculate_confidence(self, description, tags):
        """计算分析结果的置信度"""
        try:
            # 基于描述长度和标签数量计算一个简单的置信度分数
            description_score = min(len(description) / 200, 1.0)  # 假设理想描述长度为200字符
            tags_score = min(len(tags) / 5, 1.0)  # 假设理想标签数量为5个
            
            # 综合评分
            confidence = (description_score * 0.7 + tags_score * 0.3)
            
            return round(confidence, 2)
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
        
    
    def _select_voice_id(self, voice_map, narration_voice, narration_style):
        """
        根据旁白声音和风格选择合适的voice_id
        
        narration_voice: 0=无, 1=男性, 2=女性
        narration_style: 0=无, 1=慈祥, 2=文静, 3=幽默, 4=雄壮, 5=和蔼
        """
        if narration_voice == 0 or narration_style == 0:
            return None
            
        # 定义风格到声音特征的映射
        style_to_description = {
            1: ['warm', 'friendly'],  # 慈祥
            2: ['soft', 'articulate'],  # 文静
            3: ['casual', 'expressive'],  # 幽默
            4: ['intense', 'deep'],  # 雄壮
            5: ['friendly', 'warm']  # 和蔼
        }
        
        # 获取当前风格对应的描述词列表
        style_descriptions = style_to_description.get(narration_style, [])
        
        # 筛选符合性别和风格的声音
        suitable_voices = []
        for name, voice in voice_map.items():
            labels = voice.get('labels', {})
            
            # 检查性别匹配
            gender_match = (
                (narration_voice == 1 and labels.get('gender') == 'male') or
                (narration_voice == 2 and labels.get('gender') == 'female')
            )
            
            if not gender_match:
                continue
                
            # 检查风格匹配
            description = labels.get('description', '').lower()
            if any(style in description for style in style_descriptions):
                suitable_voices.append(voice['id'])
        
        # 如果找到合适的声音，返回第一个；否则返回该性别的任意声音
        if suitable_voices:
            return suitable_voices[0]
        
        # 备选：返回符合性别的第一个声音
        for voice in voice_map.values():
            labels = voice.get('labels', {})
            if ((narration_voice == 1 and labels.get('gender') == 'male') or
                (narration_voice == 2 and labels.get('gender') == 'female')):
                return voice['id']
        
        return None

    def _format_description(self, description, lang):
        """
        格式化描述文本
        """
        # 去除多余的空格和换行
        description = re.sub(r'\s+', ' ', description).strip()
        
        # 根据语言进行特定处理
        if lang == 'zh':
            # 在中文句号后添加换行
            description = re.sub(r'。', '。\n', description)
            # 移除最后一个换行符
            description = description.rstrip('\n')
            # 确保段落之间只有一个换行符
            description = re.sub(r'\n+', '\n', description)
        elif lang == 'en':
            # 在英文句号后添加换行（考虑缩写的情况）
            description = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', '\n', description)
            # 确保段落之间只有一个换行符
            description = re.sub(r'\n+', '\n', description)
        elif lang == 'ja':
            # 在日文句号后添加换行
            description = re.sub(r'。', '。\n', description)
            # 移除最后一个换行符
            description = description.rstrip('\n')
            # 确保段落之间只有一个换行符
            description = re.sub(r'\n+', '\n', description)
        
        return description.strip()
    
    def _clean_narration(self, narration, lang):
        """
        清理旁白文本
        """
        # 去除多余的空格和换行
        narration = re.sub(r'\s+', ' ', narration).strip()
        narration = narration.strip('"""\'\'\'')  # 移除可能的引号
        
        # 根据语言进行特定处理
        if lang == 'zh':
            # 确保中文标点符号正确使用
            narration = re.sub(r'[!?]', '！', narration)  # 统一感叹号
            narration = re.sub(r'\.{3,}', '……', narration)  # 统一省略号
        elif lang == 'en':
            # 确保英文标点符号正确使用
            narration = re.sub(r'\.{3,}', '...', narration)  # 统一省略号
            # 确保句子首字母大写
            narration = '. '.join(s.capitalize() for s in narration.split('. '))
        elif lang == 'ja':
            # 确保日文标点符号正确使用
            narration = re.sub(r'!', '！', narration)  # 统一感叹号
            narration = re.sub(r'\.{3,}', '……', narration)  # 统一省略号
        
        return narration.strip()

    
    