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



@blip_bp.route('/generate_caption', methods=['POST'])
def generate_caption():
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
    



@blip_bp.route('/extract_video_frames', methods=['POST'])
def extract_video_frames():
    """Extract key frames from a video file"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if 'video_url' not in data:
            return jsonify({"error": "video_url is required"}), 400
        if 'project_no' not in data:
            return jsonify({"error": "project_no is required"}), 400
            
        # Optional parameters with defaults
        frame_interval = data.get('frame_interval', 1)  # seconds
        max_frames = data.get('max_frames', 5)  # maximum number of frames
        
        # Get video handler instance
        video_handler = Factory.get_instance(VideoHandler)
        
        # Extract frames
        frame_results = video_handler.extract_key_frames(
            video_url=data['video_url'],
            project_no=data['project_no'],
            frame_interval=frame_interval,
            max_frames=max_frames
        )

        if frame_results is None:
            return jsonify({
                "success": False,
                "error": "Failed to extract video frames",
                "video_url": data['video_url']
            }), 500

        return jsonify({
            "success": True,
            "frames": frame_results,
            "video_url": data['video_url'],
            "total_frames": len(frame_results)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "video_url": data.get('video_url', '')
        }), 500
    


'''
curl -X POST \
-H "Content-Type: application/json" \
-d '{
    "video_url": "https://seeming-1322557366.cos.ap-chongqing.myqcloud.com/origin/coffee2.mp4",
    "project_no": "test1",
    "desired_frames": 20
}' \
http://workspace.featurize.cn:60048/blip/video_captions
'''
@blip_bp.route('/video_captions', methods=['POST'])
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
        
        # Get handler instances
        video_handler = Factory.get_instance(VideoHandler)
        blip_handler = Factory.get_instance(BlipHandler)
        
        # Extract frames
        frame_results = video_handler.process_keyframes(
            project_no=data['project_no'],
            video_url=data['video_url'],
            desired_frames=data['desired_frames']
        )

        if frame_results is None:
            return jsonify({
                "success": False,
                "error": "Failed to extract video frames",
                "video_url": data['video_url']
            }), 500

        # Generate captions for each frame
        captioned_frames = []
        for frame_info in frame_results:
            caption = blip_handler.generate_caption(
                image_url=frame_info['frame_url'],
                project_no=data['project_no']
            )
            
            captioned_frames.append({
                'frame_url': frame_info['frame_url'],
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