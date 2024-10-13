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

