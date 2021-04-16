import math
from typing import Tuple, Dict

from datacube.model import GridSpec
from datacube.utils.geometry import GeoBox, CRS, box


class AfricaGeobox:
    """
    generate the geobox for each tile according to the longitude ande latitude bounds.
    add origin to remove the negative coordinate
    x_new = x_old  + 181
    y_new = y_old + 77
    """

    def __init__(self, resolution: Tuple[int, int] = (-10, 10), crs: str = "epsg:6933"):
        target_crs = CRS(crs)
        self.albers_africa_N = GridSpec(
            crs=target_crs,
            tile_size=(96_000.0, 96_000.0),  # default
            resolution=resolution,
            origin=(-7392000, -17376000),
        )
        africa = box(-18, -38, 60, 30, "epsg:4326")
        self.africa_projected = africa.to_crs(crs, resolution=math.inf)

    def tile_geobox(self, tile_index: Tuple[int, int]) -> GeoBox:
        return self.albers_africa_N.tile_geobox(tile_index)

    @property
    def geobox_dict(self) -> Dict:
        return dict(self.albers_africa_N.tiles(self.africa_projected.boundingbox))
