from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import logging
import os


class COSUtil:
    def __init__(self, region='ap-chongqing', scheme='https'):
        token = None               # 如果使用永久密钥不需要填入 token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见 https://cloud.tencent.com/document/product/436/14048
        endpoint = 'seeming-1322557366.cos.ap-chongqing.myqcloud.com' # 替换为用户的 endpoint 或者 cos 全局加速域名，如果使用桶的全球加速域名，需要先开启桶的全球加速功能，请参见 https://cloud.tencent.com/document/product/436/38864
        self.bucket_name = "seeming-1322557366"  # 存储桶名称需要全局唯一
        self.secret_id = os.getenv('COS_SECRET_ID')
        self.secret_key = os.getenv('COS_SECRET_KEY')

        self.config = CosConfig(Region=region, SecretId=self.secret_id, SecretKey=self.secret_key,Token=token, Scheme=scheme)

        self.client = CosS3Client(self.config)
        
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        self.logger = logging.getLogger(__name__)

    def create_bucket(self, bucket_name):
        try:
            response = self.client.create_bucket(
                Bucket=bucket_name
            )
            print(response)
            self.logger.info(f"Bucket created successfully: {bucket_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error creating bucket: {e}")
            return False

    def list_buckets(self):
        try:
            response = self.client.list_buckets()
            print(response)
            buckets = response['Buckets']['Bucket']
            self.logger.info(f"Buckets listed successfully. Count: {len(buckets)}")
            return buckets
        except Exception as e:
            self.logger.error(f"Error listing buckets: {e}")
            return []

    def upload_file(self, bucket, local_file_path, cos_file_path):
        try:
            # Get the file size
            file_size = os.path.getsize(local_file_path)
            
            # Calculate an appropriate part size (minimum 5MB, maximum 5GB)
            part_size = max(5, min(file_size // 10000, 5120))
            
            response = self.client.upload_file(
                Bucket=bucket,
                LocalFilePath=local_file_path,
                Key=cos_file_path,
                PartSize=part_size,
                MAXThread=10,
                EnableMD5=True
            )
            
            self.logger.info(f"File uploaded successfully: {cos_file_path}")
            return response
        except Exception as e:
            self.logger.error(f"Error uploading file: {e}")
            return None

    def list_objects(self, bucket, prefix=''):
        try:
            response = self.client.list_objects(
                Bucket=bucket,
                Prefix=prefix
            )
            objects = [content['Key'] for content in response.get('Contents', [])]
            self.logger.info(f"Objects listed successfully. Count: {len(objects)}")
            return objects
        except Exception as e:
            self.logger.error(f"Error listing objects: {e}")
            return []

    def download_file(self, bucket, cos_file_path, local_file_path):
        try:
            response = self.client.download_file(
                Bucket=bucket,
                Key=cos_file_path,
                DestFilePath=local_file_path
            )
            self.logger.info(f"File downloaded successfully: {local_file_path} and response: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error downloading file: {e}")
            return None

    def delete_object(self, bucket, cos_file_path):
        try:
            response = self.client.delete_object(
                Bucket=bucket,
                Key=cos_file_path
            )
            self.logger.info(f"Object deleted successfully: {cos_file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting object: {e}")
            return False
