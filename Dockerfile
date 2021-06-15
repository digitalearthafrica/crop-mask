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
COPY testing/eastern_cropmask/results/gm_mads_two_seasons_ml_model_20210427.joblib /crop-mask/testing/eastern_cropmask/results/gm_mads_two_seasons_ml_model_20210427.joblib
COPY testing/eastern_cropmask/results/training_data/gm_mads_two_seasons_training_data_20210427.txt /crop-mask/testing/eastern_cropmask/results/training_data/gm_mads_two_seasons_training_data_20210427.txt
COPY testing/eastern_cropmask/data/Eastern.geojson /crop-mask/testing/eastern_cropmask/data/Eastern.geojson
COPY production/dea_ml/dea_ml/config/plugin_product.yaml /crop-mask/production/dea_ml/dea_ml/config/plugin_product.yaml
COPY production/dea_ml/dea_ml/config/ml_config.yaml /crop-mask/production/dea_ml/dea_ml/config/ml_config.yaml

WORKDIR /tmp
COPY --from=env_builder $py_env_path $py_env_path

RUN env && echo $PATH && pip freeze && pip check
RUN cm-pred --help
