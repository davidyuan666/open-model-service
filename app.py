
from flask import Flask
from api.image_service import image_bp
from api.audio_service import audio_bp
import os
import asyncio

'''
pip install celery redis
'''

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


app.register_blueprint(image_bp, url_prefix='/image')
app.register_blueprint(audio_bp, url_prefix='/audio')



def run_flask():
    app.run(host='0.0.0.0', port=8866, debug=True)  # Development


'''
http://sgvzncs1.cloud.lanyun.net:8866/image/
'''
# 启动应用
if __name__ == '__main__':
    run_flask()

