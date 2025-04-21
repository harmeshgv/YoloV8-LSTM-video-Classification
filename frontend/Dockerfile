FROM python:3.9.13

WORKDIR /app

COPY /frontend /app/

RUN pip install -r requirements.txt 

EXPOSE 8501

CMD ["streamlit", "run", "extract.py", "--server.port=8501", "--server.address=0.0.0.0"]