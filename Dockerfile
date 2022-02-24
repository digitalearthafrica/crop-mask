FROM continuumio/miniconda3

# Initial setup
RUN mkdir -p /conf

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && apt-get install -y build-essential

# Add in the dask configuration
COPY docker/distributed.yaml /etc/dask/distributed.yaml

# Set up conda
COPY docker/cropmask_conda.yml docker/version.txt /conf/
RUN conda update -n base -c defaults conda \
  && conda env create -f /conf/cropmask_conda.yml

# Make the environment always activate
RUN echo "conda activate cropmask" > ~/.bashrc

# Manually install Rasterio, so it works, maybe
RUN pip install rasterio --no-binary rasterio

# Check we can use rsgislib
RUN python -c "import rsgislib; print(rsgislib.__version__)" \
  && python -c "from rsgislib.segmentation import segutils" \
  && pip freeze

# # Install our Python requirements
# COPY docker/requirements.txt /conf/
# RUN pip install --upgrade pip \
#   && pip install --no-cache-dir -r /conf/requirements.txt \
#   && pip freeze

# Install the crop mask tools
ADD production/cm_tools /code
RUN pip install /code \
  && pip freeze

# Copy across region specific models, geojsons, and training data
# Eastern region:
COPY testing/eastern_cropmask/results/gm_mads_two_seasons_ml_model_20210427.joblib /eastern/gm_mads_two_seasons_ml_model_20210427.joblib
COPY testing/eastern_cropmask/results/training_data/gm_mads_two_seasons_training_data_20210427.txt /eastern/gm_mads_two_seasons_training_data_20210427.txt
COPY testing/eastern_cropmask/data/Eastern.geojson /eastern/Eastern.geojson
# Western region:
COPY testing/western_cropmask/results/western_ml_model_20210609.joblib /western/western_ml_model_20210609.joblib
COPY testing/western_cropmask/results/training_data/western_training_data_20210609.txt /western/western_training_data_20210609.txt
COPY testing/western_cropmask/data/Western.geojson /western/Western.geojson
# Northern region:
COPY testing/northern_cropmask/results/northern_ml_model_20210803.joblib /northern/northern_ml_model_20210803.joblib
COPY testing/northern_cropmask/results/training_data/northern_training_data_20210803.txt /northern/northern_training_data_20210803.txt
COPY testing/northern_cropmask/data/Northern.geojson /northern/Northern.geojson
# Sahel region:
COPY testing/sahel_cropmask/results/sahel_ml_model_20211110.joblib /sahel/sahel_ml_model_20211110.joblib
COPY testing/sahel_cropmask/results/training_data/sahel_training_data_20211110.txt /sahel/sahel_training_data_20211110.txt
COPY testing/sahel_cropmask/data/Sahel.geojson /sahel/Sahel.geojson
# Southern region:
COPY testing/southern_cropmask/results/southern_ml_model_20211108.joblib /southern/southern_ml_model_20211108.joblib
COPY testing/southern_cropmask/results/training_data/southern_training_data_20211108.txt /southern/southern_training_data_20211108.txt
COPY testing/southern_cropmask/data/Southern.geojson /southern/Southern.geojson
# South eastern region:
COPY testing/southeast_cropmask/results/southeast_ml_model_20220222.joblib /southeast/southeast_ml_model_20220222.joblib
COPY testing/southeast_cropmask/results/training_data/southeast_training_data_20220222.txt /southeast/southeast_training_data_20220222.txt
COPY testing/southeast_cropmask/data/Southeast.geojson /southeast/Southeast.geojson

# Execution environment
WORKDIR /code
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cropmask"]

# Prove it works!
RUN python -c "import rsgislib; print(rsgislib.__version__)" \
  && python -c "from rsgislib.segmentation import segutils" \
  && cat /conf/version.txt \
  && cm-task --help \
  && odc-stats --version
