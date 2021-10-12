FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get -y update && apt-get install -y --no-install-recommends \ 
ffmpeg unzip wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x download_models.sh

CMD ["sh", "-c", "./download_models.sh && python3 app.py 0.0.0.0 8000"]
