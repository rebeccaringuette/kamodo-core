FROM python:3.8.5-slim-buster
RUN apt-get update && apt-get install -y automake autogen build-essential make cmake gcc g++ libssl-dev
RUN pip install --user pkgconfig==1.5.1 Cython==0.29.21
RUN pip install --user pycapnp==1.1.0

RUN pip install cryptography

RUN apt-get install -y git

RUN git clone --single-branch --branch rpc https://github.com/EnsembleGovServices/kamodo-core.git

RUN pip install -e kamodo-core
