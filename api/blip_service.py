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

# 设置蓝图
blip_bp = Blueprint('blip', __name__)


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