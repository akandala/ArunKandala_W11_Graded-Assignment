FROM python:3.10
WORKDIR /code
COPY requirements.txt .
RUN pip3.10 install -r requirements.txt
COPY / .
CMD ["python","app.py"]
