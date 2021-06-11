FROM opendatacube/geobase:wheels-3.0.4 as env_builder
ARG py_env_path=/env

ENV LC_ALL=C.UTF-8

# Install our Python requirements
COPY docker/requirements.txt docker/version.txt /conf/

RUN cat /conf/version.txt && \
  env-build-tool new /conf/requirements.txt ${py_env_path}

RUN /env/bin/pip install --upgrade --extra-index-url="https://packages.dea.ga.gov.au" rsgislib

# Install the crop mask tools
ADD production/dea_ml /tmp/dea_ml
RUN /env/bin/pip install \
  --extra-index-url="https://packages.dea.ga.gov.au" /tmp/dea_ml && \
  rm -rf /tmp/dea_ml

# Below is the actual image that does the running
FROM opendatacube/geobase:runner
ARG py_env_path=/env

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="${py_env_path}/bin:${PATH}" \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Add in the dask configuration
COPY docker/distributed.yaml /etc/dask/distributed.yaml

ADD docker/apt-run.txt /tmp/
RUN apt-get update \
    && sed 's/#.*//' /tmp/apt-run.txt | xargs apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Read configure from docker image folder
COPY production /crop-mask/production
COPY testing /crop-mask/testing

WORKDIR /tmp
COPY --from=env_builder $py_env_path $py_env_path

RUN env && echo $PATH && pip freeze && pip check
RUN cm-pred --help
