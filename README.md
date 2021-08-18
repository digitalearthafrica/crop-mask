<img align="centre" src="figs/Github_banner.jpg" width="101%">

# Digital Earth Africa Continental Cropland Mask

## Background

A central focus for governing bodies in Africa is the need to secure the necessary food sources to support their populations. It has been estimated that the current production of crops will need to double by 2050 to meet future needs for food production (GIS Geography, 2018).  Higher level crop-based products that can assist with managing food insecurity, such as cropping watering intensities, crop types, or crop productivity, require as a starting point precise and accurate cropland extent maps indicating where cropland occurs. Current cropland extent maps are either inaccurate, have too coarse spatial resolutions, or are not updated regularly. An accurate, high-resolution, and regularly updated cropland area map for the African continent is therefore recognised as a gap in the current crop monitoring services.

## Description

The notebooks and scripts in this repository build cropland extent (crop/non-crop) maps for each of the seven simplified Agro-Ecological Zones (AEZ) shown in figure 1. The process for building models and generating test classifications for each AEZ are contained in their namesake folders within the `testing/` directory (e.g. `testing/eastern_cropmask`).

For generating production runs of the crop mask, refer to the `production/` directory.

The `pre-post_processing` folder contains notebooks used for various miscellaneous tasks.  These include things like generating randomly placed polygons for training data collection, pre and post processing of reference data, generating ancillary datasets that are used as feature layers in the ML models, etc.



_figure 1: Simplified Agro-Ecological Zones. Seperate cropland models are produced for each region._
<img align="centre" src="figs/AEZs.png" width="35%">

The cropland maps all share the following specifications:

* Developed using Sentinel-2 satellite imagery
* Have a spatial resolution of 10 metres
* Developed using data from 2019
* **Cropland definition**:
    * `A piece of land of minimum 0.16 ha that is sowed/planted and harvest-able at least once within the 12 months after the sowing/planting date.`
    * This definition will exclude grasslands and perennial crops which can be difficult for satellite imagery to differentiate from natural vegetation.

***

## Additional information

**License:** The code in this notebook is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
Digital Earth Africa data is licensed under the [Creative Commons by Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) license.

**Contact:** If you need assistance, please post a question on the [Open Data Cube Slack channel](http://slack.opendatacube.org/) or on the [GIS Stack Exchange](https://gis.stackexchange.com/questions/ask?tags=open-data-cube) using the `open-data-cube` tag (you can view previously asked questions [here](https://gis.stackexchange.com/questions/tagged/open-data-cube)).
If you would like to report an issue with this notebook, you can file one on [Github](https://github.com/digitalearthafrica/crop-mask/issues).
