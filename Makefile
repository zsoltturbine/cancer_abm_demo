FROM python:3.11.9-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["solara", "run", "app.py", "--host=0.0.0.0", "--port=7860"]
