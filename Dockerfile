FROM python:3.10-slim

WORKDIR /app

# Instalamos dependencias de sistema mínimas para seguridad y redes
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]