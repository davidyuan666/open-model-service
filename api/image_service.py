
import json
import os
import logging
from flask import Blueprint, request, jsonify, current_app
from handlers.clip_handler import CLIPHandler
from api.handler_factory import Factory


# 设置蓝图
image_bp = Blueprint('image', __name__)

@image_bp.record_once
def on_load(state):
    # 初始化 CLIPHandler 并存储在 app 的配置中
    print('loading once')
    clip_handler = Factory.get_instance(CLIPHandler)
    state.app.config['CLIP_HANDLER'] = clip_handler


@image_bp.route('/clip/generate', methods=['POST'])
async def clip_generate():
    """
    Processes a cooking query and returns the result from the cooking handler.
    """
    if 'text' not in request.json:
        return jsonify({"message": "No text provided"}), 400
    
    query = request.json['text']

    try:

        clip_handler = current_app.config['CLIP_HANDLER']
        result = clip_handler.encode_text(query)
        return jsonify({"message":result}),200
    except Exception as e:
        logging.error(f"Error in process_cooking_query: {str(e)}")
        return jsonify({"message": "An error occurred while processing the cooking query"}), 500

    