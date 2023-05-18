#!/usr/bin/bash
mkdir -p vad
wget -nc -P vad https://raw.githubusercontent.com/snakers4/silero-vad/master/files/silero_vad.onnx 

mkdir -p paraformer
wget -nc -P paraformer https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28/resolve/main/model.onnx
wget -nc -P paraformer https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28/resolve/main/tokens.txt

git lfs install
#git clone https://huggingface.co/THUDM/chatglm-6b

mkdir -p test_audios
wget https://aphid.fireside.fm/d/1437767933/a05075d5-4f3a-45ac-afff-580f795c5d77/8f71f4fa-0c6d-4758-95d2-15b72095ca4e.mp3 -O test_audios/e114.mp3
