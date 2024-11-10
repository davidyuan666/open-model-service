import utils
import torch
from pathlib import Path
from models.blip import blip_decoder
import requests
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify

app = Flask(__name__)

# 全局变量存储模型和设备
model = None
device = None

def init_model(gpu_id=0):
    """初始化模型"""
    global model, device
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 检查并创建checkpoints目录
    if not Path("checkpoints").is_dir():
        print("checkpoint directory not found.")
        utils.create_dir("checkpoints")

    # 下载模型检查点
    if not Path("checkpoints/model_large_caption.pth").is_file():
        utils.download_checkpoint()

    print("Checkpoint loading...")
    model = blip_decoder(
        pretrained="./checkpoints/model_large_caption.pth", 
        image_size=384, 
        vit="large"
    )
    model.eval()
    model = model.to(device)
    print(f"Model loaded to {device}")

def load_image_from_url(image_url):
    """从URL加载图片"""
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        raise Exception(f"Error loading image from URL: {str(e)}")

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    """处理POST请求，生成图片描述"""
    try:
        # 检查请求数据
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if 'image_url' not in data:
            return jsonify({"error": "image_url is required"}), 400

        # 加载图片
        image = load_image_from_url(data['image_url'])
        
        # 预处理图片
        transformed_image = utils.prep_single_image(image, device)

        # 生成描述
        with torch.no_grad():
            caption = model.generate(
                transformed_image, 
                sample=False, 
                num_beams=3, 
                max_length=20, 
                min_length=5
            )

        return jsonify({
            "success": True,
            "caption": caption[0],
            "image_url": data['image_url']
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # 初始化模型
    init_model()
    # 启动Flask服务
    app.run(host='0.0.0.0', port=5000)