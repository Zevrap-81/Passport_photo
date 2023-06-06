apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \ 
    python3-tk \
    git-all \

apt-get clean && rm -rf /var/lib/apt/lists/*

pip install --no-cache-dir -r requirements.txt

python3 -m ipykernel install --sys-prefix