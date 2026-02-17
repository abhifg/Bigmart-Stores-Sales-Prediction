FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN apt-get update -y && \
    apt-get install -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 80

CMD ["python3", "app.py"]
