
import json
import os
import logging
from flask import Blueprint, request, jsonify, current_app,render_template
from handlers.clip_handler import CLIPHandler
from handlers.sd_handler import StableDiffusionHandler
from api.handler_factory import Factory
import base64
from io import BytesIO
import time
import base64
import io
from PIL import Image
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 设置蓝图
image_bp = Blueprint('image', __name__)


@image_bp.record_once
def on_load(state):
    # 初始化 CLIPHandler 并存储在 app 的配置中
    print('loading once')
    clip_handler = Factory.get_instance(CLIPHandler)
    state.app.config['CLIP_HANDLER'] = clip_handler
    sd_handler = Factory.get_instance(StableDiffusionHandler)
    state.app.config['SD_HANDLER'] = sd_handler


@image_bp.route('/')
async def index():
    return render_template('index.html')


@image_bp.route('/search')
async def search():
    return render_template('search.html')


@image_bp.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(os.getcwd(),UPLOAD_FOLDER, filename)
        file.save(file_path)
        return jsonify({"image_name": filename}), 200
    return jsonify({"error": "File type not allowed"}), 400


@image_bp.route('/clip/search', methods=['POST'])
async def clip_search():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    query = data['text']
    image_name = data['image']
    language = data.get('language', 'chn')
    querys = []
    querys.append(query)


    print(f'image name: {image_name}')
    image_path = os.path.join(os.getcwd(),UPLOAD_FOLDER, image_name)

    if not os.path.exists(image_path):
        return jsonify({"error": f"Image file not found at {image_path}"}), 404


    try:
        clip_handler = current_app.config['CLIP_HANDLER']
        if clip_handler is None:
            return jsonify({"error": "CLIP handler not initialized"}), 500
        
        
        # Encode text
        if language == 'eng':
            text_features = clip_handler.encode_text_eng(query)
        elif language == 'chn':
            text_features = clip_handler.encode_text_chn(querys)
            print(text_features)
        else:
            return jsonify({"error": "Unsupported language"}), 400
        
        # Encode image
        if language == 'eng':
            img_features = clip_handler.encode_image_eng(image_path)
        elif language == 'chn':
            img_features = clip_handler.encode_image_chn(image_path)
            print(img_features)
        else:
            return jsonify({"error": "Unsupported language"}), 400
        
        # Calculate similarity
        similarity_score = clip_handler.calculate_similarity(img_features, text_features)
        
        return jsonify({
            "similarity_score": similarity_score
        }), 200
    except FileNotFoundError:
        return jsonify({"error": "Image file not found"}), 404
    except Exception as e:
        print(f"Error in clip_search: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the query {e}"}), 500


@image_bp.route('/clip/encode_text', methods=['POST'])
async def encode_text():
    """
    Encodes the input text and returns the feature vector.
    """
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400

    query = request.json['text']
    language = request.json.get('language', 'chn')  # Default to English if not specified

    try:
        clip_handler = current_app.config['CLIP_HANDLER']
        if language == 'eng':
            text_features = clip_handler.encode_text_eng(query)
        elif language == 'chn':
            text_features = clip_handler.encode_text_chn(query)
        else:
            return jsonify({"error": "Unsupported language"}), 400
        return jsonify({"text_features": text_features.tolist()}), 200
    except Exception as e:
        logging.error(f"Error in encode_text: {str(e)}")
        return jsonify({"error": "An error occurred while encoding the text"}), 500

@image_bp.route('/clip/encode_image', methods=['POST'])
async def encode_image():
    """
    Encodes the input image and returns the feature vector.
    """
    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400

    image_path = request.json['image']
    language = request.json.get('language', 'chn')  # Default to English if not specified

    try:
        clip_handler = current_app.config['CLIP_HANDLER']
        if language == 'eng':
            img_features = clip_handler.encode_image_eng(image_path)
        elif language == 'chn':
            img_features = clip_handler.encode_image_chn(image_path)
        else:
            return jsonify({"error": "Unsupported language"}), 400
        return jsonify({"image_features": img_features.tolist()}), 200
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return jsonify({"error": "Image file not found"}), 404
    except Exception as e:
        logging.error(f"Error in encode_image: {str(e)}")
        return jsonify({"error": "An error occurred while encoding the image"}), 500
    


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
        
        start_time = time.time()
        image = sd_handler.generate_image(query)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Convert the image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            "message": "Image generated successfully",
            "image": img_str,
            "generation_time": f"{generation_time:.2f} seconds"
        }), 200
    except Exception as e:
        logging.error(f"Error in sd_generate: {str(e)}")
        return jsonify({"error": "An error occurred while generating the image"}), 500
    
