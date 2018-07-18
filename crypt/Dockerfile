FROM python:2.7

RUN pip install pycrypto

ADD . /app

WORKDIR /app

ENTRYPOINT ["python", "rsa.py"]
