sudo apt update
sudo apt install ffmpeg
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install -r requirements.txt
# featurize port export 5000