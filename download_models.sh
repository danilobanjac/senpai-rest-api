#!/bin/bash

MODEL_DIR="model_en"

if [ -d "$MODEL_DIR" ]; then
  echo "Models already downloaded, skipping..."
else
echo "Downloading models..."
{
  wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.21.zip
  unzip vosk-model-en-us-0.21.zip
  rm -rf vosk-model-en-us-0.21.zip
  mv vosk-model-en-us-0.21 model_en

  wget https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip
  unzip vosk-model-de-0.21.zip
  rm -rf vosk-model-de-0.21.zip
  mv vosk-model-de-0.21 model_de

  python3 -m spacy download en_core_web_trf
  python3 -m spacy download de_dep_news_trf
} &> /dev/null
echo "Models downloaded successfully!"
fi
