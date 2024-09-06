FROM python:3.10

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt --default-timeout=100 future

EXPOSE 8501

CMD ["streamlit","run","app.py"]

#docker
# docker images

# docker build -t sakthi49/pdfchat:v1.0 .
# docker build -t sakthi49/pdfchat:latest .

# docker login
# docker push sakthi49/pdfchat:latest .