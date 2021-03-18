# Database building

The original doc of [db_setup](https://datacube-core.readthedocs.io/en/latest/ops/db_setup.html).

In project root run ```docker-compose up```. Three containers will be created.

- postgresql
- localstack
- stats

1. ```createdb datacube```
   or
   ```createdb -h <hostname> -U <username> datacube```

2. ```datacube -v system init```

3. Python class ```Datacube``` will look for a configuration file in `````~/.datacube.conf`````
which will be edited like sample below

```bash
[datacube]
 db_hostname: 191.168.0.19
 db_username: postgres
 db_password: opendatacubepassword
```

4. add product into datacube

```bash
datacube product add crop_mask_eastern.yaml
```

### Notes

The product yaml will has following fields. And the ```metadata``` will be picked up as the signature in the code.

```yaml
---
name: crop_mask_eastern
description: Estern Africa region crop mask prediction based on features
   of s2_l2a statistics gm_tmad, plus NDVI, LAI, MNDWI, rainfall, slope
metadata_type: eo3

license: CC-BY-4.0

metadata:
   product:
      name: crop_mask_eastern

storage:
   crs: epsg:6933
   resolution:
      x: 20
      y: -20
   tile_size:
      x: 96000
      y: 96000

measurements:
   - name: mask
     aliases: ['crop_mask', 'MASK']
     dtype: uint8
     nodata: 255
     units: '1'

   - name: prob
     aliases: ['crop_prob', 'PROB']
     dtype: uint8
     nodata: 255
     units: '1'
```
