FROM continuumio/miniconda3

# Initial setup
RUN mkdir -p /conf

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update \
  && apt-get install -y \
  build-essential \
  awscli \
  wget

# Add in the dask configuration
COPY docker/distributed.yaml /etc/dask/distributed.yaml

# Set up conda
COPY docker/cropmask_conda.yml docker/version.txt /conf/
RUN conda update -n base -c defaults conda \
  && conda env create -f /conf/cropmask_conda.yml

# Make the environment always activate
RUN echo "conda activate cropmask" > ~/.bashrc

# Pip requirements (rasterio and Tools)
COPY docker/requirements.txt /conf/
RUN pip install -r /conf/requirements.txt

# Check we can use rsgislib
RUN python -c "import rsgislib; print(rsgislib.__version__)" \
  && python -c "from rsgislib.segmentation import segutils" \
  && pip freeze

# Install the crop mask tools
ADD production/cm_tools /code
RUN pip install /code \
  && pip freeze

# Copy production data files into the code folder
ADD data /code/data

# Execution environment
WORKDIR /code
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cropmask"]

# Prove it works! Smoke tests and so on... really should have real code tests.
RUN python -c "import rsgislib; print(rsgislib.__version__)" \
  && python -c "from rsgislib.segmentation import segutils" \
  && cat /conf/version.txt \
  && cm-task --help \
  && odc-stats --version
