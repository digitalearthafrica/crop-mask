# import pytest
# from mock import patch
# from dea_ml.plugins.gm_ml_pred import PredGMS2
# from odc.dscache.tools.tiling import GRIDS
# import yaml
# import os.path as osp
#
#
# @pytest.fixture
# def dummy_pred_gms2():
#     dirname = osp.dirname(__file__)
#     with open(osp.join(dirname, "ml_config.yaml")) as fh:
#         args = yaml.safe_load(fh)
#     return PredGMS2(**args)
#
#
# @pytest.mark.skip(reason="integration check the input_data methods only")
# def test_pred_gms2_pred_input(dummy_pred_gms2):
#     from dask.distributed import Client
#
#     client = Client(processes=False)  # noqa F841
#     with patch("odc.stats.model.Task") as mock:
#         instance = mock.return_value
#         instance.tile_index = (220, 77)
#         instance.geobox = GRIDS["africa_10"].tile_geobox(instance.tile_index)
#         ds = dummy_pred_gms2.input_data(instance)
#         print(ds)
#         assert ds
