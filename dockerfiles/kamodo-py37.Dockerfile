# docker build -t asherp/kamodo -f API.Dockerfile .
FROM condaforge/miniforge3
# FROM continuumio/miniconda3:latest
LABEL maintainer "Asher Pembroke <apembroke@gmail.com>"

RUN conda install python=3.7

# RUN conda install jupyter
RUN pip install antlr4-python3-runtime


# # Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
# ENV TINI_VERSION v0.6.0
# ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# RUN chmod +x /usr/bin/tini
# ENTRYPOINT ["/usr/bin/tini", "--"]

# need to pin this version for api
RUN pip install sympy==1.5.1

# Keep plotly at lower api
RUN pip install plotly==4.7.1

# kaleido for generating static plots
RUN pip install kaleido

# capnproto pip version
RUN conda install gcc cmake make cxx-compiler
RUN pip install pkgconfig cython

# # install release
RUN  wget https://capnproto.org/capnproto-c++-0.9.1.tar.gz
RUN  tar zxf capnproto-c++-0.9.1.tar.gz
WORKDIR capnproto-c++-0.9.1
RUN  ./configure 
RUN  make -j6 check
RUN  make install

# RUN pip install pycapnp --install-option "--force-bundled-libcapnp"

# # capnp install
# RUN conda install -c conda-forge pycapnp

# RUN conda install -c conda-forge clang
# # RUN conda install -c conda-forge gcc cmake make cxx-compiler
# RUN pip install pkgconfig cython
# RUN conda install -c conda-forge autoconf automake libtool
# # libtool looks for sed in /usr/bin/sed, so we soft link it
# RUN ln -sf /bin/sed /usr/bin/sed 
# RUN ln -sf /bin/grep /usr/bin/grep

# RUN pip install --no-binary :all: --install-option "--force-system-libcapnp" pycapnp

# # Install latest kamodo
# ADD . /kamodo

# # RUN git clone https://github.com/asherp/kamodo.git
# RUN pip install -e kamodo

# RUN conda install jupyter
# RUN pip install jupytext

# WORKDIR kamodo

# # CMD ["kamodo-serve"]

# CMD ["jupyter", "notebook", "./docs/notebooks", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

# #####
# # For Jupyter notebook interaction, use:
# #	docker run -p 8888:8888 dezeeuw/kamodo
# # For command line interaction, use:
# #	docker run -it dezeeuw/kamodo /bin/bash
# #   -above, with current working directory mounted in container, use
# #	docker run -it --mount type=bind,source="$(pwd)",destination=/local,consistency=cached  dezeeuw/kamodo /bin/bash
# #   -above, with persistent disk space, use
# #	docker run -it --mount source=kamododisk,target=/kdisk dezeeuw/kamodo /bin/bash
# #
# # Persistent disk space command
# #	docker volume create kamododisk
# #