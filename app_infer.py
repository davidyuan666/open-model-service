import utils
import torch
from pathlib import Path
from models.blip import blip_decoder
import requests
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from utils.tencent_cos_util import COSOperationsUtil
from urllib.parse import urlparse
import os

app = Flask(__name__)

# 全局变量存储模型和设备
model = None
device = None

'''
https://github.com/salesforce/LAVIS/tree/main/projects/blip2

featurize port export 5000
https://docs.featurize.cn/docs/manual/port-exporting  (端口转发，最多支持10个)

workspace.featurize.cn:44768

'''
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

def download_video_from_cos(cos_video_url, project_no):
        try:
            cos_util = COSOperationsUtil()
            
            parsed_url = urlparse(cos_video_url)

            object_key = parsed_url.path.lstrip('/')
            
            # Extract the original filename from the object_key
            original_filename = os.path.basename(object_key)
            
            video_local_path = os.path.join(os.getcwd(), 'temp', project_no, original_filename)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(video_local_path), exist_ok=True)

            cos_util.download_file(cos_util.bucket_name, object_key, video_local_path)


            return video_local_path
        
        except Exception as e:
            print(f"Error getting video from COS URL {cos_video_url}: {str(e)}")
            return None
        

def load_image_from_cos(image_url, project_no):
    """从COS加载图片
    
    Args:
        image_url (str): COS中图片的URL
        project_no (str): 项目编号
        
    Returns:
        PIL.Image: 加载的图片对象
        
    Raises:
        Exception: 当图片加载失败时抛出异常
    """
    try:
        # 从COS下载文件到本地
        local_path = download_video_from_cos(image_url, project_no)
        if not local_path:
            raise Exception("Failed to download file from COS")
            
        # 打开并转换图片
        try:
            image = Image.open(local_path).convert('RGB')
            return image
        except Exception as e:
            raise Exception(f"Error opening image file: {str(e)}")
        finally:
            # 可选：清理临时文件
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {local_path}: {str(e)}")
                
    except Exception as e:
        raise Exception(f"Error loading image from COS URL {image_url}: {str(e)}")


@app.route('/test', methods=['GET'])
def test_connection():
    """测试API连接和模型状态"""
    try:
        return jsonify({
            "success": True,
            "status": "API is running",
            "model_loaded": model is not None,
            "device": str(device)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500



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
        image = load_image_from_cos(data['image_url'],data['project_no'])
        
        # 预处理图片
        transformed_image = utils.prep_images([image], device)

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