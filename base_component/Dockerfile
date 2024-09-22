FROM python:3.9

WORKDIR /src

COPY requirements.txt ./
COPY ./modules /src/modules

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/Simone-Albero/nids-framework.git /src/external_library
RUN pip install --no-cache-dir /src/external_library/nids_framework