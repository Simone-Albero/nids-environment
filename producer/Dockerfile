FROM python:3.9

WORKDIR /src

COPY requirements.txt ./
COPY main.py ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 6000

CMD ["python", "main.py"]