FROM python:3.9-slim

WORKDIR /app

COPY ./frontend/ /app/

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt 

EXPOSE 8501

CMD ["streamlit", "run", "extract.py", "--server.port=8501", "--server.address=0.0.0.0"]