
# soundfile/Dockerfile
FROM python:3.8.5

ENV API_KEY = "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"

WORKDIR home\caellwyn\projects\soundslike

COPY . .

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y libsndfile1-dev

RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]