import math
from typing import Tuple, Dict

from datacube.model import GridSpec
from datacube.utils.geometry import GeoBox, CRS, box


class AfricaGeobox:
    """
    generate the geobox for each tile according to the longitude ande latitude bounds.
    """

    def __init__(self, resolution: Tuple[int, int] = (-20, 20), crs: str = "epsg:6933"):
        target_crs = CRS(crs)
        self.albers_africa_N = GridSpec(
            crs=target_crs,
            tile_size=(96_000.0, 96_000.0),  # default
            resolution=resolution,
        )
        africa = box(-18, -38, 60, 30, "epsg:4326")
        self.africa_projected = africa.to_crs(crs, resolution=math.inf)

    def tile_geobox(self, tile_index: Tuple[int, int]) -> GeoBox:
        return self.albers_africa_N.tile_geobox(tile_index)

    @property
    def geobox_dict(self) -> Dict:
        return dict(self.albers_africa_N.tiles(self.africa_projected.boundingbox))
