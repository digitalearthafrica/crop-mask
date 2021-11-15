ARG py_env_path=/env
ARG V_BASE=3.3.0

FROM opendatacube/geobase-builder:${V_BASE} as env_builder
ENV LC_ALL=C.UTF-8

# Install our Python requirements
COPY docker/requirements.txt docker/version.txt docker/constraints.txt /conf/

RUN cat /conf/version.txt && \
  env-build-tool new /conf/requirements.txt /conf/constraints.txt ${py_env_path}

# Install the crop mask tools
ADD production/cm_tools /tmp/cm_tools
RUN /env/bin/pip install \
  --extra-index-url="https://packages.dea.ga.gov.au" /tmp/cm_tools && \
  rm -rf /tmp/cm_tools

# Below is the actual image that does the running
FROM opendatacube/geobase-runner:${V_BASE}
ARG py_env_path=/env

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="${py_env_path}/bin:${PATH}" \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update \
    && apt-get install software-properties-common -y \
    && apt-get upgrade -y
    
# Add in the dask configuration
COPY docker/distributed.yaml /etc/dask/distributed.yaml
ADD docker/apt-run.txt /tmp/
RUN apt-get update \
    && sed 's/#.*//' /tmp/apt-run.txt | xargs apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy across region specific models, geojsons, and training data
#Eastern region:
COPY testing/eastern_cropmask/results/gm_mads_two_seasons_ml_model_20210427.joblib /eastern/gm_mads_two_seasons_ml_model_20210427.joblib
COPY testing/eastern_cropmask/results/training_data/gm_mads_two_seasons_training_data_20210427.txt /eastern/gm_mads_two_seasons_training_data_20210427.txt
COPY testing/eastern_cropmask/data/Eastern.geojson /eastern/Eastern.geojson
#Western region:
COPY testing/western_cropmask/results/western_ml_model_20210609.joblib /western/western_ml_model_20210609.joblib
COPY testing/western_cropmask/results/training_data/western_training_data_20210609.txt /western/western_training_data_20210609.txt
COPY testing/western_cropmask/data/Western.geojson /western/Western.geojson
#Northern region:
COPY testing/northern_cropmask/results/northern_ml_model_20210803.joblib /northern/northern_ml_model_20210803.joblib
COPY testing/northern_cropmask/results/training_data/northern_training_data_20210803.txt /northern/northern_training_data_20210803.txt
COPY testing/northern_cropmask/data/Northern.geojson /northern/Northern.geojson
#Sahel region:
COPY testing/sahel_cropmask/results/sahel_ml_model_20211110.joblib /sahel/sahel_ml_model_20211110.joblib
COPY testing/sahel_cropmask/results/training_data/sahel_training_data_20211110.txt /sahel/sahel_training_data_20211110.txt
COPY testing/sahel_cropmask/data/Sahel.geojson /sahel/Sahel.geojson
#Southern region:
COPY testing/southern_cropmask/results/southern_ml_model_20211108.joblib /southern/southern_ml_model_20211108.joblib
COPY testing/southern_cropmask/results/training_data/southern_training_data_20211108.txt /southern/southern_training_data_20211108.txt
COPY testing/southern_cropmask/data/Southern.geojson /southern/Southern.geojson


WORKDIR /tmp
COPY --from=env_builder $py_env_path $py_env_path

RUN env && echo $PATH && pip freeze && pip check
RUN cm-task --help
