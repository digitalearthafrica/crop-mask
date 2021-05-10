ARG V_BASE=-3.0.4
ARG py_env_path=/env

FROM opendatacube/geobase:wheels${V_BASE} as env_builder
ARG py_env_path

RUN mkdir -p /conf

# Some libs assume presence of Cython and numpy in the environment, so build
# minimal env with those first, then extend further
COPY requirements0.txt /conf/
RUN echo "Building python environment: stage 1" \
  && env-build-tool new /conf/requirements0.txt ${py_env_path}

COPY requirements.txt /conf/
RUN echo "Building python environment: stage 2" \
  && env-build-tool extend /conf/requirements.txt ${py_env_path}

ENV PATH=${py_env_path}/bin:$PATH

COPY requirements-odc-static.txt /conf/
RUN echo "Adding odc-dependencies (static)" \
  && env-build-tool extend /conf/requirements-odc-static.txt ${py_env_path}

FROM opendatacube/geobase:runner${V_BASE}

ENV DEBIAN_FRONTEND=noninteractive

#Install ffmpeg static binary
RUN apt-get update -y \
  && apt-get install -y --fix-missing --no-install-recommends xz-utils curl \
  && rm -rf /var/lib/apt/lists/* \
  && curl -s -L https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.2.2-amd64-static.tar.xz > ffmpeg.tar.xz \
  && echo "c1c7ec5180523db214a7f6f59ea737e8e582570cb884c3ab0c583da4f5165e27" ffmpeg.tar.xz | sha256sum -c - \
  && tar xvJ --strip-components=1 -C /tmp < ffmpeg.tar.xz \
  && mkdir -p /usr/local/bin/ \
  && mv /tmp/ffmpeg /usr/local/bin/ffmpeg \
  && rm -rf /tmp/* ffmpeg.tar.xz

RUN apt-get update -y \
&& apt-get install -y --fix-missing --no-install-recommends \
  # developer convenience
  postgresql-client-10 \
  postgresql-10 \
  less \
  wget \
  curl \
  vim \
  tmux \
  htop \
  fish \
  tig \
  git \
  jq \
  xz-utils \
  zip \
  unzip \
  file \
  time \
  openssh-client \
  graphviz \
  sudo \
  iproute2 \
  iputils-ping \
  net-tools \
  simpleproxy \
  rsync \
  libtiff-tools \
# rgsislib dependencies
  libcgal13 \
  libmuparser2v5 \
  libgsl23 \
  libboost-system1.65.1 \
  libboost-filesystem1.65.1 \
  libboost-date-time1.65.1 \
# for cython to work need compilers
  build-essential \
# for pyRAT install or something
  libfftw3-dev \
  liblapack-dev \
&& rm -rf /var/lib/apt/lists/*

# Install Tini
COPY --from=env_builder /bin/tini /bin/tini

ARG nb_user=cm_worker
ARG nb_uid=1000
ARG nb_gid=100

RUN useradd -m -s /bin/bash -N -g $nb_gid -u $nb_uid $nb_user

# Copy python env
ARG py_env_path
COPY --chown=1000:100 --from=env_builder $py_env_path $py_env_path

ENV LC_ALL=C.UTF-8
ENV SHELL=bash
# Put `/usr/local/bin` before env to allow overrides in there
ENV PATH=/usr/local/bin:${py_env_path}/bin:$PATH

# Use `--build-arg PURGE_CACHE_ODC=$(date)` to force re-installation of ODC dependencies
# without the need to "edit" requirements-odc.txt
ARG PURGE_CACHE_ODC=N
COPY requirements-odc.txt /conf/
USER $nb_user
RUN echo "Adding odc-dependencies" \
  && if [ "${PURGE_CACHE_ODC}" != "N" ]; then echo "Forced docker to purge cache for this step!"; fi \
  && ${py_env_path}/bin/pip install \
  --no-cache-dir \
  --requirement=/conf/requirements-odc.txt

# install crop-mask-dependencies dea-tools and dea-ml through github http url
COPY dea-tools-ml.txt /conf/
USER $nb_user
RUN echo "Adding crop-mask-dependencies" \
  && ${py_env_path}/bin/pip install \
  --no-cache-dir \
  --requirement=/conf/dea-tools-ml.txt

WORKDIR "/home/$nb_user"

ARG WITH_SUDO="no"
RUN if [ "$WITH_SUDO" = "yes" ]; then \
  echo "${nb_user} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
;fi

USER $nb_user
