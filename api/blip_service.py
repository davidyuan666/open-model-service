import json
import os
from flask import Blueprint, request, jsonify, current_app, render_template, send_file, url_for
from api.handler_factory import Factory
import base64
from io import BytesIO
from PIL import Image
from werkzeug.utils import secure_filename
import requests
import uuid
import urllib.parse
from requests.exceptions import RequestException
import numpy as np
import utils
import requests
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from urllib.parse import urlparse
import os
from handlers.blip_handler import BlipHandler
from api.handler_factory import Factory
from handlers.video_handler import VideoHandler

# 设置蓝图
blip_bp = Blueprint('blip', __name__)


# 在文件顶部添加缓存字典
_video_cache = {}  # 格式: {video_url: {'local_path': path, 'last_accessed': timestamp}}



'''
curl -X POST -H "Content-Type: application/json" -d '{"image_url":"https://seeming-1322557366.cos.ap-chongqing.myqcloud.com/test01/frames/frame_d09c1e27-5cc7-4fe6-be3c-bff119396f3b.jpg", "project_no":"test1"}' http://workspace.featurize.cn:60048/blip/generate_caption
'''

@blip_bp.route('/test', methods=['GET'])
def test_connection():
    """测试API连接和模型状态"""
    try:
        return jsonify({
            "success": True,
            "status": "API is running"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


'''
生成图片描述
'''
@blip_bp.route('/image/generate_caption', methods=['POST'])
def generate_image_caption():
    """处理POST请求，生成图片描述"""
    try:
        # 检查请求数据
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if 'image_url' not in data:
            return jsonify({"error": "image_url is required"}), 400
        
        blip_handler = Factory.get_instance(BlipHandler)
        
        caption = blip_handler.generate_caption(data['image_url'],data['project_no'])

        return jsonify({
            "success": True,
            "caption": caption,
            "image_url": data['image_url']
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "caption": str(e),
            "image_url": data['image_url']
        }), 500
    



'''
Step1:合成所有的子视频
'''
@blip_bp.route('/merge_videos', methods=['POST'])
def merge_videos():
    """Merge multiple input videos into a single video"""
    temp_files = []  # Track files to clean up
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if 'video_urls' not in data:
            return jsonify({"error": "video_urls is required"}), 400
        if 'project_no' not in data:
            return jsonify({"error": "project_no is required"}), 400
        if not isinstance(data['video_urls'], list):
            return jsonify({"error": "video_urls must be a list"}), 400

        # Get handler instance
        video_handler = Factory.get_instance(VideoHandler)
        
        # Download and merge videos
        merge_response = video_handler.merge_input_videos(
            video_urls=data['video_urls'],
            project_no=data['project_no']
        )

        if merge_response is None:
            return jsonify({
                "success": False,
                "error": "Failed to merge videos",
                "project_no": data['project_no']
            }), 500

        # Upload merged video to COS
        upload_response = video_handler.upload_video_to_cos(
            local_video_path=merge_response['local_merged_video_path'],
            project_no=merge_response['project_no']
        )

        _video_cache[upload_response['video_url']] = {
            "local_path": merge_response['local_merged_video_path']
        }

        return jsonify({
            "success": True,
            "merged_video_url": upload_response['video_url'],
            "project_no": upload_response['project_no']
        })

    except Exception as e:
        print(f'error: {str(e)}')
        return jsonify({
            "success": False,
            "error": str(e),
            "project_no": data.get('project_no', '')
        }), 500

    finally:
        # Clean up temporary files after successful upload
        for temp_file in temp_files:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as cleanup_error:
                print(f"Failed to clean up file {temp_file}: {str(cleanup_error)}")



                
'''
Step2:提取视频中的所有帧，并生成每个帧的描述
curl -X POST \
-H "Content-Type: application/json" \
-d '{
    "video_url": "https://seeming-1322557366.cos.ap-chongqing.myqcloud.com/origin/coffee2.mp4",
    "project_no": "test1",
    "desired_frames": 20
}' \
http://workspace.featurize.cn:60048/blip/video/generate_captions
'''
@blip_bp.route('/video/generate_captions', methods=['POST'])
def generate_video_captions():
    """Extract frames from video and generate captions for each frame"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if 'video_url' not in data:
            return jsonify({"error": "video_url is required"}), 400
        if 'project_no' not in data:
            return jsonify({"error": "project_no is required"}), 400
        if 'desired_frames' not in data:
            return jsonify({"error": "desired_frames is required"}), 400
        if 'interval_seconds' not in data:
            return jsonify({"error": "interval_seconds is required"}),400
        
        # Get handler instances
        video_handler = Factory.get_instance(VideoHandler)
        blip_handler = Factory.get_instance(BlipHandler)
        frame_results = None
        if data['video_url'] in _video_cache:
            local_video_path = _video_cache[data['video_url']]['local_path']
        else:
            local_video_path = None

        if local_video_path is None:
            frame_results = video_handler.process_all_frames(
                project_no=data['project_no'],
                video_url=data['video_url'],
                local_video_path=None,
                interval_seconds = data['interval_seconds']
            )
        else:
            frame_results = video_handler.process_all_frames(
                project_no=data['project_no'],
                video_url=data['video_url'],
                local_video_path=local_video_path,
                interval_seconds = data['interval_seconds']
            )


        if frame_results is None:
            return jsonify({
                "success": False,
                "error": "Failed to extract video frames",
                "video_url": data['video_url']
            }), 500

        # Generate captions for each frame
        captioned_frames = []
        for frame_info in frame_results['frames']:
            caption = blip_handler.generate_caption_by_local_path(
                local_frame_path=frame_info['local_frame_path']
            )
            
            captioned_frames.append({
                'local_frame_path': frame_info['local_frame_path'],
                'timestamp': frame_info['timestamp'],
                'caption': caption
            })

        return jsonify({
            "success": True,
            "video_url": data['video_url'],
            "total_frames": len(captioned_frames),
            "frames": captioned_frames
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "video_url": data.get('video_url', '')
        }), 500
    

'''
Step3:合成提取之后的视频
'''
@blip_bp.route('/synthesize_video', methods=['POST'])
def synthesize_video():
    """Synthesize multiple video clips into a single video"""
    temp_files = []  # Track files to clean up
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if 'selected_clips_directory' not in data:
            return jsonify({"error": "selected_clips_directory is required"}), 400
        if 'project_no' not in data:
            return jsonify({"error": "project_no is required"}), 400

        # Get handler instances
        video_handler = Factory.get_instance(VideoHandler)
        
        # Synthesize video
        merged_video_path, clip_infos = video_handler.synthesize_video(
            selected_clips_directory=data['selected_clips_directory'],
            project_no=data['project_no']
        )

        if merged_video_path:
            temp_files.append(merged_video_path)

        
        # Add clip paths to temp_files for cleanup
        clip_paths = [info['path'] for info in clip_infos]
        temp_files.extend(clip_paths)


        # Upload individual clips to COS
        clip_urls = video_handler.upload_clips_to_cos(
            clip_paths=clip_paths,
            project_no=data['project_no']
        )
        
        # Upload merged video to COS
        merged_video_url = video_handler.upload_video_to_cos(
            video_path=merged_video_path,
            project_no=data['project_no']
        )

        # Combine clip URLs with their durations
        clip_details = []
        for i, clip_info in enumerate(clip_infos):
            clip_details.append({
                'url': clip_urls[i]['url'],
                'duration': clip_info['duration']
            })


        # 格式化打印日志
        print("\n=== Video Processing Results ===")
        print("Clips:")
        for i, clip in enumerate(clip_details, 1):
            print(f"  Clip {i}:")
            print(f"    URL: {clip['url']}")
            print(f"    Duration: {clip['duration']} seconds")
        print("\nMerged Video:")
        print(f"  URL: {merged_video_url}")
        print("============================\n")
        

        
        return jsonify({
            "success": True,
            "merged_video_url": merged_video_url,
            "project_no": data['project_no'],
            "clip_urls": clip_details
        })



    except Exception as e:
        print(f'error: {str(e)}')
        return jsonify({
            "success": False,
            "error": str(e),
            "project_no": data.get('project_no', ''),
            "selected_clips_directory": data.get('selected_clips_directory', '')
        }), 500

    finally:
        # Clean up temporary files after successful upload
        for temp_file in temp_files:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as cleanup_error:
                print(f"Failed to clean up file {temp_file}: {str(cleanup_error)}")


