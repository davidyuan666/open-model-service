import json
import os
import logging
from flask import Blueprint, request, jsonify, current_app, render_template, send_file, url_for
from handlers.clip_handler import CLIPHandler
from handlers.sd_handler import StableDiffusionHandler
from api.handler_factory import Factory
import base64
from io import BytesIO
import time
import tempfile
from PIL import Image
from werkzeug.utils import secure_filename
import requests
import uuid
import urllib.parse
from requests.exceptions import RequestException
import numpy as np


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def load_and_encode_image(clip_handler, image_path, language):
    try:
        if image_path.startswith(('http://', 'https://')):
            if 'http://sgvzncs1.cloud.lanyun.net:8866/' in image_path:
                # Handle local file
                image_name = image_path.split('/')[-1]
                local_path = os.path.join(os.getcwd(), 'uploads', image_name)
                if not os.path.exists(local_path):
                    raise FileNotFoundError(f"Local file not found: {local_path}")
                img = Image.open(local_path)
            else:
                # Handle remote file
                response = requests.get(image_path, timeout=10)  # 10 seconds timeout
                response.raise_for_status()  # Raises an HTTPError for bad responses
                img = Image.open(BytesIO(response.content))
        else:
            # Load image from local path
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Local file not found: {image_path}")
            img = Image.open(image_path)

        # Convert image to RGB if it's not
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            img.save(temp_file.name)
            temp_path = temp_file.name

        # Encode image
        if language == 'eng':
            features = clip_handler.encode_image_eng(temp_path)
        elif language == 'chn':
            features = clip_handler.encode_image_chn(temp_path)
        else:
            raise ValueError("Unsupported language")

        # Remove temporary file
        os.unlink(temp_path)

        return features

    except RequestException as e:
        # Handle network-related errors
        raise Exception(f"Error fetching image from URL: {str(e)}")
    except FileNotFoundError as e:
        # Handle file not found errors
        raise Exception(str(e))
    except IOError as e:
        # Handle image opening and processing errors
        raise Exception(f"Error processing image: {str(e)}")
    except Exception as e:
        # Handle any other unexpected errors
        raise Exception(f"Unexpected error in load_and_encode_image: {str(e)}")


def generate_encoded_url(filename):
    # Generate a unique identifier
    unique_id = str(uuid.uuid4())
    # Encode only the unique ID
    encoded_id = base64.urlsafe_b64encode(unique_id.encode()).decode()
    # Return a URL-friendly string with both the encoded ID and the original filename
    return f"{encoded_id}/{filename}"



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


@image_bp.route('/generate')
async def generate():
    return render_template('generate.html')


@image_bp.route('/compare')
async def compare():
    return render_template('compare.html')


@image_bp.route('/search')
async def search():
    return render_template('search.html')


@image_bp.route('/clip/compare_images', methods=['POST'])
async def compare_images():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'image1' not in data or 'image2' not in data:
        return jsonify({"error": "Both image1 and image2 must be provided"}), 400

    image1_path = data['image1']
    image2_path = data['image2']
    language = data.get('language', 'chn')

    try:
        clip_handler = current_app.config['CLIP_HANDLER']
        if clip_handler is None:
            return jsonify({"error": "CLIP handler not initialized"}), 500

        # Load and encode images
        img1_features = load_and_encode_image(clip_handler, image1_path, language)
        img2_features = load_and_encode_image(clip_handler, image2_path, language)

        # Calculate similarity
        similarity_score = clip_handler.calculate_similarity(img1_features, img2_features)

        return jsonify({
            "similarity_score": similarity_score
        }), 200
    except Exception as e:
        print(f"Error in compare_images: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the query: {str(e)}"}), 500



@image_bp.route('/clip/compare_image_text', methods=['POST'])
async def clip_compare_image_text():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    query = data['text']
    image_path = data['image']
    language = data.get('language', 'chn')
    queries = [query]

    try:
        clip_handler = current_app.config['CLIP_HANDLER']
        if clip_handler is None:
            return jsonify({"error": "CLIP handler not initialized"}), 500
        
        # Encode text
        if language == 'eng':
            text_features = clip_handler.encode_text_eng(query)
        elif language == 'chn':
            text_features = clip_handler.encode_text_chn(queries)
        else:
            return jsonify({"error": "Unsupported language"}), 400
        
        # Encode image
        img_features = load_and_encode_image(clip_handler, image_path, language)
        
        # Calculate similarity
        similarity_score = clip_handler.calculate_similarity(img_features, text_features)
        
        return jsonify({
            "similarity_score": float(similarity_score)
        }), 200
    except FileNotFoundError as e:
        return jsonify({"error": f"Image file not found: {str(e)}"}), 404
    except Exception as e:
        print(f"Error in clip_compare_image_text: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the query: {str(e)}"}), 500
    


@image_bp.route('/clip/compare_weighted_images_text', methods=['POST'])
async def clip_compare_weighted_images_text():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    if 'images' not in data or not isinstance(data['images'], list):
        return jsonify({"error": "No images provided or images is not a list"}), 400

    if 'weights' not in data or not isinstance(data['weights'], list):
        return jsonify({"error": "No weights provided or weights is not a list"}), 400

    if len(data['images']) != len(data['weights']):
        return jsonify({"error": "Number of images and weights must be the same"}), 400

    query = data['text']
    image_paths = data['images']
    weights = data['weights']
    language = data.get('language', 'chn')
    queries = [query]

    try:
        clip_handler = current_app.config['CLIP_HANDLER']
        if clip_handler is None:
            return jsonify({"error": "CLIP handler not initialized"}), 500
        
        # Encode text
        if language == 'eng':
            text_features = clip_handler.encode_text_eng(query)
        elif language == 'chn':
            text_features = clip_handler.encode_text_chn(queries)
        else:
            return jsonify({"error": "Unsupported language"}), 400
        
        # Encode images and calculate weighted features
        weighted_features = None
        for image_path, weight in zip(image_paths, weights):
            img_features = load_and_encode_image(clip_handler, image_path, language)
            if weighted_features is None:
                weighted_features = img_features * weight
            else:
                weighted_features += img_features * weight
        
        # Normalize weighted features
        weighted_features /= np.sum(weights)
        
        # Calculate similarity
        similarity_score = clip_handler.calculate_similarity(weighted_features, text_features)
        
        return jsonify({
            "similarity_score": float(similarity_score),
            "weighted_features": weighted_features.tolist()
        }), 200
    except FileNotFoundError as e:
        return jsonify({"error": f"Image file not found: {str(e)}"}), 404
    except Exception as e:
        print(f"Error in clip_compare_weighted_images_text: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the query: {str(e)}"}), 500




@image_bp.route('/clip/find_similar_images', methods=['POST'])
async def find_similar_images():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'input_image' not in data:
        return jsonify({"error": "No input image provided"}), 400
    
    if 'candidate_images' not in data or not isinstance(data['candidate_images'], list):
        return jsonify({"error": "No candidate images provided or it's not a list"}), 400

    threshold = data.get('threshold', 0.5)
    language = data.get('language', 'chn')

    input_image = data['input_image']
    candidate_images = data['candidate_images']

    try:
        clip_handler = current_app.config['CLIP_HANDLER']
        if clip_handler is None:
            return jsonify({"error": "CLIP handler not initialized"}), 500
        
        # Encode input image
        input_features = load_and_encode_image(clip_handler, input_image, language)
        
        similar_images = []
        for candidate in candidate_images:
            candidate_features = load_and_encode_image(clip_handler, candidate, language)
            similarity = clip_handler.calculate_similarity(input_features, candidate_features)
            if similarity > threshold:
                similar_images.append({"image": candidate, "similarity": float(similarity)})
        
        return jsonify({
            "similar_images": similar_images
        }), 200
    except Exception as e:
        print(f"Error in find_similar_images: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the query: {str(e)}"}), 500
    



@image_bp.route('/clip/find_images_by_text', methods=['POST'])
async def find_images_by_text():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    if 'candidate_images' not in data or not isinstance(data['candidate_images'], list):
        return jsonify({"error": "No candidate images provided or it's not a list"}), 400

    threshold = data.get('threshold', 0.5)
    language = data.get('language', 'chn')

    text = data['text']
    candidate_images = data['candidate_images']
    queries = [text]

    try:
        clip_handler = current_app.config['CLIP_HANDLER']
        if clip_handler is None:
            return jsonify({"error": "CLIP handler not initialized"}), 500
        
        # Encode text
        if language == 'eng':
            text_features = clip_handler.encode_text_eng(text)
        elif language == 'chn':
            text_features = clip_handler.encode_text_chn(queries)
        else:
            return jsonify({"error": "Unsupported language"}), 400
        
        similar_images = []
        for candidate in candidate_images:
            img_features = load_and_encode_image(clip_handler, candidate, language)
            similarity = clip_handler.calculate_similarity(text_features, img_features)
            if similarity > threshold:
                similar_images.append({"image": candidate, "similarity": float(similarity)})
        
        return jsonify({
            "similar_images": similar_images
        }), 200
    except Exception as e:
        print(f"Error in find_images_by_text: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the query: {str(e)}"}), 500




@image_bp.route('/delete_image', methods=['POST'])
def delete_image():
    data = request.json
    if 'image_name' not in data:
        return jsonify({"error": "No image name provided"}), 400
    
    image_name = data['image_name']
    file_path = os.path.join(current_app.root_path, UPLOAD_FOLDER, image_name)
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"message": f"Image {image_name} deleted successfully"}), 200
        else:
            return jsonify({"error": f"Image {image_name} not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Error deleting image: {str(e)}"}), 500
    




# Add this route to serve images
@image_bp.route('/images/<encoded_id>/<filename>')
def serve_image(encoded_id, filename):
    try:
        # Decode the unique ID
        unique_id = base64.urlsafe_b64decode(encoded_id.encode()).decode()
        
        # Ensure the filename is secure
        filename = secure_filename(filename)
        
        # Construct the full path
        file_path = os.path.join(current_app.root_path, UPLOAD_FOLDER, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            current_app.logger.error(f"Image not found: {file_path}")
            return jsonify({"error": "Image not found"}), 404
        
        # Serve the file
        return send_file(file_path)
    except Exception as e:
        current_app.logger.error(f"Error serving image: {str(e)}")
        return jsonify({"error": f"Error serving image: {str(e)}"}), 500
    


# Modify the upload_image function
@image_bp.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.root_path, UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Generate encoded URL
        encoded_url = generate_encoded_url(filename)
        image_url = url_for('image.serve_image', 
                            encoded_id=encoded_url.split('/')[0], 
                            filename=filename, 
                            _external=True, 
                            _scheme=request.scheme)
        
        # 解析生成的 URL 并添加端口号
        parsed_url = list(urllib.parse.urlparse(image_url))
        parsed_url[1] = f"{parsed_url[1].split(':')[0]}:8866"  # 添加端口号 8866
        image_url_with_port = urllib.parse.urlunparse(parsed_url)
        
        return jsonify({
            "image_name": filename,
            "image_url": image_url_with_port
        }), 200
    return jsonify({"error": "File type not allowed"}), 400



@image_bp.route('/list_uploaded_images', methods=['GET'])
def list_uploaded_images():
    upload_folder = os.path.join(current_app.root_path, UPLOAD_FOLDER)
    files = []
    for filename in os.listdir(upload_folder):
        if allowed_file(filename):
            encoded_url = generate_encoded_url(filename)
            image_url = url_for('image.serve_image', 
                                encoded_id=encoded_url.split('/')[0], 
                                filename=filename, 
                                _external=True, 
                                _scheme=request.scheme)
            
            # 解析生成的 URL 并添加端口号
            parsed_url = list(urllib.parse.urlparse(image_url))
            parsed_url[1] = f"{parsed_url[1].split(':')[0]}:8866"  # 添加端口号 8866
            image_url_with_port = urllib.parse.urlunparse(parsed_url)
            
            files.append({
                'name': filename,
                'url': image_url_with_port
            })
    return jsonify(files)




@image_bp.route('/clip/image_image_search', methods=['POST'])
async def clip_image_image_search():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'image1' not in data:
        return jsonify({"error": "No first image provided"}), 400
    
    if 'image2' not in data:
        return jsonify({"error": "No second image provided"}), 400

    image1_name = data['image1']
    image2_name = data['image2']
    language = data.get('language', 'chn')

    image1_path = os.path.join(os.getcwd(), UPLOAD_FOLDER, image1_name)
    image2_path = os.path.join(os.getcwd(), UPLOAD_FOLDER, image2_name)

    if not os.path.exists(image1_path):
        return jsonify({"error": f"Image file not found at {image1_path}"}), 404
    if not os.path.exists(image2_path):
        return jsonify({"error": f"Image file not found at {image2_path}"}), 404

    try:
        clip_handler = current_app.config['CLIP_HANDLER']
        if clip_handler is None:
            return jsonify({"error": "CLIP handler not initialized"}), 500
        
        # Encode images
        if language == 'eng':
            img1_features = clip_handler.encode_image_eng(image1_path)
            img2_features = clip_handler.encode_image_eng(image2_path)
        elif language == 'chn':
            img1_features = clip_handler.encode_image_chn(image1_path)
            img2_features = clip_handler.encode_image_chn(image2_path)
        else:
            return jsonify({"error": "Unsupported language"}), 400
        
        # Calculate similarity
        similarity_score = clip_handler.calculate_similarity(img1_features, img2_features)
        
        return jsonify({
            "similarity_score": similarity_score
        }), 200
    except FileNotFoundError as e:
        return jsonify({"error": f"Image file not found: {str(e)}"}), 404
    except Exception as e:
        print(f"Error in clip_search: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the query: {str(e)}"}), 500






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
    
