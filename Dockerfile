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

# Copy across region specific models, geojsons, and td
#Eastern region:
COPY testing/eastern_cropmask/results/gm_mads_two_seasons_ml_model_20210427.joblib /eastern/gm_mads_two_seasons_ml_model_20210427.joblib
COPY testing/eastern_cropmask/results/training_data/gm_mads_two_seasons_training_data_20210427.txt /eastern/gm_mads_two_seasons_training_data_20210427.txt
COPY testing/eastern_cropmask/data/Eastern.geojson /eastern/Eastern.geojson
#Western region:
COPY testing/western_cropmask/results/western_ml_model_20210609.joblib /western/western_ml_model_20210609.joblib
COPY testing/western_cropmask/results/training_data/western_training_data_20210609.txt /western/western_training_data_20210609.txt
COPY testing/western_cropmask/data/Western.geojson /western/Western.geojson

WORKDIR /tmp
COPY --from=env_builder $py_env_path $py_env_path

RUN env && echo $PATH && pip freeze && pip check
RUN cm-pred --help
