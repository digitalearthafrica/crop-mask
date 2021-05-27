ARG py_env_path=/env
ARG V_BASE=3.3.0

FROM opendatacube/geobase-builder:${V_BASE} as env_builder
ARG py_env_path

ENV LC_ALL=C.UTF-8

# Install our Python requirements
RUN mkdir -p /conf
COPY docker/requirements.txt docker/constraints.txt docker/version.txt /conf/
RUN cat /conf/version.txt && \
  env-build-tool new /conf/requirements.txt /conf/constraints.txt ${py_env_path} \
  && rm -rf /root/.cache/pip \
  && echo done

ADD production/dea_ml /tmp/dea_ml

RUN pip install /tmp/dea_ml && \
    rm -rf /tmp/dea_ml

# Below is the actual image that does the running
FROM opendatacube/geobase-runner:${V_BASE}
ARG py_env_path

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

WORKDIR /tmp
COPY --from=env_builder $py_env_path $py_env_path

RUN env && echo $PATH && pip freeze && pip check
RUN odc-stats --help
