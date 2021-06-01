import logging
from typing import Tuple, Dict, Optional, Any

import xarray as xr
from dea_ml.core.feature_layer import gm_mads_two_seasons_prediction
from dea_ml.core.post_processing import post_processing
from dea_ml.core.predict_from_feature import predict_with_model
from dea_ml.helpers.io import read_joblib
from odc.stats import _plugins
from odc.stats.model import Task, StatsPluginInterface

_log = logging.getLogger(__name__)


class PredGMS2(StatsPluginInterface):
    """
    Prediction from GeoMAD
    task run with the template
    datakube-apps/src/develop/workspaces/deafrica-dev/processing/06_stats_2019_semiannual_gm.yaml
    """

    source_product = "gm_s2_semiannual"
    target_product = "crop_mask_eastern"

    def __init__(
        self,
        urls: Dict[str, Any],
        rename_dict: Dict[str, str],
        bands: Optional[Tuple] = None,
    ):
        # target band to be saved
        self.urls = urls
        self.rename_dict = rename_dict
        self.bands = bands if bands else ("mask", "prob", "filtered")

    @property
    def measurements(self) -> Tuple[str, ...]:
        return self.bands

    def input_data(self, task: Task) -> xr.Dataset:
        """
        assemble the input data and do prediction here.
        This method works as pipeline
        """
        # create the features
        measurements = [
            "blue",
            "green",
            "red",
            "nir",
            "swir_1",
            "swir_2",
            "red_edge_1",
            "red_edge_2",
            "red_edge_3",
            "bcdev",
            "edev",
            "sdev",
        ]

        pred_input_data = gm_mads_two_seasons_prediction(task, measurements, self.urls)

        if not pred_input_data:
            return None
        # read in model
        model = read_joblib(self.urls["model"])

        # run predictions
        predicted = predict_with_model(
            model=model, data=pred_input_data, chunk_size={}, td_url=self.urls["td"]
        )

        # rechunk on the way out
        return predicted.chunk({"x": -1, "y": -1})

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        return post_processing(xx, self.urls)


_plugins.register("pred-gm-s2", PredGMS2)
