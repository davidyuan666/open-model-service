
import json
import os
import logging
from flask import Blueprint, request, jsonify, current_app,render_template
# from handlers.clip_handler import CLIPHandler
from handlers.sd_handler import StableDiffusionHandler
from api.handler_factory import Factory
import base64
from io import BytesIO

# 设置蓝图
image_bp = Blueprint('image', __name__)

@image_bp.record_once
def on_load(state):
    # 初始化 CLIPHandler 并存储在 app 的配置中
    print('loading once')
    # clip_handler = Factory.get_instance(CLIPHandler)
    # state.app.config['CLIP_HANDLER'] = clip_handler
    sd_handler = Factory.get_instance(StableDiffusionHandler)
    state.app.config['SD_HANDLER'] = sd_handler


@image_bp.route('/')
def index():
    return render_template('index.html')


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

    


@image_bp.route('/sd/generate', methods=['POST'])
async def sd_generate():
    """
    Generates an image based on the provided text prompt and returns the image as a base64 encoded string.
    """
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400
    
    query = request.json['text']

    try:
        sd_handler = current_app.config['SD_HANDLER']
        image = sd_handler.generate_image(query)
        
        # Convert the image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            "message": "Image generated successfully",
            "image": img_str
        }), 200
    except Exception as e:
        logging.error(f"Error in sd_generate: {str(e)}")
        return jsonify({"error": "An error occurred while generating the image"}), 500
    
