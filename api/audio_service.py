import json
import os
import logging
from flask import Blueprint, request, jsonify, current_app, render_template, send_file, url_for
from handlers.whisper_handler import WhisperHandler
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
audio_bp = Blueprint('audio', __name__)


@audio_bp.record_once
def on_load(state):
    # 初始化 CLIPHandler 并存储在 app 的配置中
    print('loading once')
    whisper_handler = Factory.get_instance(WhisperHandler)
    state.app.config['WHISPER_HANDLER'] = whisper_handler


@audio_bp.route('/whisper/transcribe', methods=['POST'])
def whisper_transcribe():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'audios' not in data:
        return jsonify({"error": "Audio paths must be provided"}), 400

    audio_paths = data.get('audios')
    language = data.get('language', 'chn')  # Default to Chinese if not specified

    try:
        whisper_handler = current_app.config['WHISPER_HANDLER']
        if whisper_handler is None:
            return jsonify({"error": "Whisper handler not initialized"}), 500

        # Transcribe audio(s)
        if isinstance(audio_paths, list):
            results = whisper_handler.transcribe(audio_paths, batch_size=len(audio_paths))
        else:
            results = whisper_handler.transcribe(audio_paths)

        # Process results
        if isinstance(results, list):
            transcriptions = [result['text'] for result in results]
        else:
            transcriptions = [results['text']]

        return jsonify({
            "transcriptions": transcriptions
        }), 200
    except Exception as e:
        print(f"Error in whisper_transcribe: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the audio: {str(e)}"}), 500

