
import json
import os
import logging
from flask import Blueprint, request, jsonify
from handlers.clip_handler import CLIPHandler


# 设置蓝图
image_bp = Blueprint('image', __name__)


@image_bp.route('/generate/embedding', methods=['POST'])
async def generate_embedding():
    """
    Processes a cooking query and returns the result from the cooking handler.
    """
    if 'text' not in request.json:
        return jsonify({"message": "No text provided"}), 400
    
    query = request.json['text']

    try:

        return jsonify({}),200
    except Exception as e:
        logging.error(f"Error in process_cooking_query: {str(e)}")
        return jsonify({"message": "An error occurred while processing the cooking query"}), 500

    