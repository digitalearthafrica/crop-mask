import math
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List
from uuid import UUID

import pystac
from datacube import Datacube
from datacube.model import Dataset, DatasetType
from datacube.utils import changes
from datacube.utils.geometry import GeoBox
from odc.index.stac import stac_transform
from odc.stats.model import (
    format_datetime,
    OutputProduct,
    TileIdx_xy,
    DateTimeRange,
)


class StacIntoDc:
    """
    use the method in odc-stas s3-to-dc
    copy from dc_tools for testing purposes only
    stac json do  stac_transform into the EO3
    from_metadata_stream into dataset
    process the prediction results into datacube. Generalized to all ad hoc products into indexing
    1. generate the metadata stac format json into s3 or local
    2. upload the predicttion tifs onto s3 allow with stac json but with band names ['MASK', 'PROB']
    3. add or update index into datacube
    """

    @staticmethod
    def index_update_dataset(
        dc: Datacube, datasets: Tuple[dict, str], update: bool, allow_unsafe: bool
    ) -> Tuple[int, int]:
        """
        copy from dc_tools for testing purposes only
        stac json do  stac_transform into the EO3
        from_metadata_stream into dataset
        @param dc:
        @param datasets: dataset, uri as in the for loop below, dataset: datacube.model.Dataset
        @param update:
        @param allow_unsafe:
        @return:
        """
        ds_added = 0
        ds_failed = 0

        for dataset, uri in datasets:
            # datacube.model.Dataset
            if uri is not None:
                if dataset is not None:
                    if update:
                        updates: Dict[Tuple, Any] = {}
                        if allow_unsafe:
                            updates = {tuple(): changes.allow_any}
                        dc.index.datasets.update(dataset, updates_allowed=updates)
                    else:
                        ds_added += 1
                        dc.index.datasets.add(dataset)
                else:
                    ds_failed += 1
            else:
                ds_failed += 1

        return ds_added, ds_failed

    @staticmethod
    def to_dc_dataset(
        dc: Datacube,
        rendered: Dict[str, Any],
        ds_type: Optional[DatasetType] = None,
        transform: Callable = stac_transform,
        product_name: str = "crop_mask",
    ) -> Dataset:
        """ "
        Stac transformed
        """
        if not ds_type:
            ds_type = dict((d.name, d) for d in dc.index.datasets.types.get_all())[
                product_name
            ]
        return Dataset(ds_type, transform(rendered))

    @staticmethod
    def render_metadata(
        product: OutputProduct,
        geobox: GeoBox,
        tile_index: TileIdx_xy,
        time_range: DateTimeRange,
        uuid: UUID,
        paths: Dict[str, str],
        metadata_path: str,
        processing_dt: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Put together STAC metadata document for the output from the task info.
        """
        if processing_dt is None:
            processing_dt = datetime.utcnow()

        region_code = product.region_code(tile_index)
        inputs: List[str] = []

        properties: Dict[str, Any] = deepcopy(product.properties)
        properties["dtr:start_datetime"] = format_datetime(time_range.start)
        properties["dtr:end_datetime"] = format_datetime(time_range.end)
        properties["odc:processing_datetime"] = format_datetime(
            processing_dt, timespec="seconds"
        )
        properties["odc:region_code"] = region_code
        properties["odc:lineage"] = dict(inputs=inputs)
        properties["odc:product"] = product.name

        geobox_wgs84 = geobox.extent.to_crs(
            "epsg:4326", resolution=math.inf, wrapdateline=True
        )
        bbox = geobox_wgs84.boundingbox

        item = pystac.Item(
            id=str(uuid),
            geometry=geobox_wgs84.json,
            bbox=[bbox.left, bbox.bottom, bbox.right, bbox.top],
            datetime=time_range.start.replace(tzinfo=timezone.utc),
            properties=properties,
        )

        # Enable the Projection extension
        item.ext.enable("projection")
        item.ext.projection.epsg = geobox.crs.epsg

        # Add all the assets
        for band, path in paths.items():
            asset = pystac.Asset(
                href=path,
                media_type="image/tiff; application=geotiff",
                roles=["data"],
                title=band,
            )
            item.add_asset(band, asset)

            item.ext.projection.set_transform(geobox.transform, asset=asset)
            item.ext.projection.set_shape(geobox.shape, asset=asset)

        # Add links
        item.links.append(
            pystac.Link(
                rel="product_overview",
                media_type="application/json",
                target=product.href,
            )
        )
        item.links.append(
            pystac.Link(
                rel="self",
                media_type="application/json",
                target=metadata_path,
            )
        )

        return item.to_dict()
