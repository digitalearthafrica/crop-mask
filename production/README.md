![Banner](../figs/Github_banner.jpg)

# Digital Earth Africa Continental Cropland Mask - Production

The code base here provides all the methods necessary for running the crop mask machine learning pipeline using AWS's [Kubernetes](https://kubernetes.io/) platform. The methods rely on the Open Data Cube's [Statistician](https://github.com/opendatacube/odc-stats) library for orchestrating the machine learning predicitions on AWS's cloud infrastructure. [Argo](https://argo.digitalearth.africa/workflows?limit=500) is used for deploying the code on kubernetes. [Argo](https://argoproj.github.io/workflows/) is a tool that simplifies orchestrating parallel job on kubernetes.

## How to build and install the crop-mask tools ( `cm_tools` ) library

In the folder `cm_tools/` , run the following shell command:

``` bash
pip install cm_tools

```

## Local testing of production analysis code

The ODC-statistician plugin that does the core analysis can be tested within DE Africa's Sandbox environment using the notebook [1_test_plugin.ipynb](1_test_plugin.ipynb).

* The ODC-stats plugin is called [PredGMS2](cm_tools/cm_tools/gm_ml_pred.py)
* The two primary functions that this plugin references are in the [feature_layer](cm_tools/cm_tools/feature_layer.py) and [post_processing](cm_tools/cm_tools/post_processing.py) scripts.
* A yaml is required to configure the plugin, e.g. [config_western](cm_tools/cm_tools/config/config_western.yaml)

## List of Production Models for each AEZ

Within each testing folder a number of the earlier iteration models are retained. To reduce confusion, a list of the models used to create the production crop masks are listed below (these can also be found in the `Dockerfile` ).  The date the training data and models were created identifies each iteration of model:

* Eastern: `testing/eastern_cropmask/results/gm_mads_two_seasons_ml_model_20210427.joblib`
* Western:  `testing/western_cropmask/results/western_ml_model_20210609.joblib`
* Northern: `testing/northern_cropmask/results/northern_ml_model_20210803.joblib`
* Sahel: `testing/sahel_cropmask/results/sahel_ml_model_20211110.joblib`
* Southern: `testing/southern_cropmask/results/southern_ml_model_20211108.joblib`
* South East: `testing/southeast_cropmask/results/southeast_ml_model_20220222.joblib`
* Central: `testing/central_cropmask/results/central_ml_model_20220304.joblib`

## Running production code

 > Note, the following contains an example of running the code on the production EKS, the workflow should first be tested in DEV-EKS.

* DEV Cluster: `deafrica-dev-eks`
* PROD Cluster: `deafrica-prod-af-eks`

The steps to create a large scale cropland extent map using K8s/Argo and the ML-methods described in this repo are as follows:

01. Ensure the `config_<region>` yaml is correct for the region of Africa you are running. i.e. one of these files [here](https://github.com/digitalearthafrica/crop-mask/tree/main/production/cm_tools/cm_tools/config)

02. Ensure the [Dockerfile](../Dockerfile) contains all the right files and libraries. If you alter the `Dockerfile` or the `cm_tools` code base you need to rebuild the image, this can triggered by changing the version number [here](../docker/version.txt) and creating a pull request.

03. Login to DE Africa's [Argo-production](https://argo.digitalearth.africa/workflows?limit=500) workspace (or [Argo-dev](https://argo.dev.digitalearth.africa/workflows?limit=500)), click on `submit workflow`, and use the drop-down box to select the `stats-crop-mask-processing` template.  The yaml file which creates this workflow is located [here](https://github.com/digitalearthafrica/datakube-apps/blob/main/workspaces/deafrica-prod-af/processing/argo/workflow-templates/stats-crop-mask.yaml) in the [datakube-apps](https://github.com/digitalearthafrica/datakube-apps) repo.

04. Change the parameters in the workflows that are saved on [datakube-apps](https://github.com/digitalearthafrica/datakube-apps/blob/main/workspaces/deafrica-prod-af/processing/argo/once/stats_crop_mask_2019_central.yaml) to align with the job you're running.
    * There are eight jobs for each of the eight regions. These have been templated, so to run again, edit each of the yaml files and copy and paste into the Argo workflow creation dialog to execute it
    * Note that each region has a different version, for example, the Central region is `version: 1.1.4`, whereas Eastern is `version: 1.1.0`. This is done to allow overlapping tiles between regions, and enables you to load tiles from a single region, or from multiple regions.

05. To monitor the batch run you can use:

    * [Production CPU, memory, SQS monitoring](https://mgmt.digitalearth.africa/d/CropMaskMetrics/crop-mask-annual)
    * [Dev CPU, memory, SQS monitoring](https://mgmt.dev.digitalearth.africa/d/CropMaskMetrics/crop-mask-annual)

06. To check the logs of any pod, you can click on one of the pods that displays in Argo after you hit submit and then click the `logs` button

07. Once the batch job has completed, you can follow the instructions in `2_Indexing_results_into_datacube.ipynb` to index the crop-mask into the datacube.

### Other useful run notes

* To list tiles in a s3 bucket; useful to know if results have been successfully written to disk

``` bash
aws s3 ls s3://deafrica-data-dev-af/crop_mask_western/
```

* To sync (copy) results in a s3 bucket to your local machine

``` bash
aws s3 sync s3://deafrica-data-dev-af/crop_mask_western/ crop_mask_western
```

* If doing test runs, and you wish delete test geotifs from the dev bucket

``` bash
  aws s3 rm --recursive s3://deafrica-data-dev-af/crop_mask_western --dryrun
```

---

## Additional information

**License:** The code in this notebook is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
Digital Earth Africa data is licensed under the [Creative Commons by Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) license.

**Contact:** If you need assistance, please post a question on the [Open Data Cube Slack channel](http://slack.opendatacube.org/) or on the [GIS Stack Exchange](https://gis.stackexchange.com/questions/ask?tags=open-data-cube) using the `open-data-cube` tag (you can view previously asked questions [here](https://gis.stackexchange.com/questions/tagged/open-data-cube)).
If you would like to report an issue with this notebook, you can file one on [Github](https://github.com/digitalearthafrica/crop-mask/issues).
