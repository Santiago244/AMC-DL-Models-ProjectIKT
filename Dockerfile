FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY data/splits/ ./data/splits/

CMD ["python", "-m", "src.train", "--config", "configs/cnn_baseline.yaml"]
