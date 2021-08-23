<img align="centre" src="../figs/Github_banner.jpg" width="101%">

#  Digital Earth Africa Continental Cropland Mask - Production

The code base here provides all the methods necessary for running the crop-mask machine learning pipeline using AWS's [Kubernetes](https://kubernetes.io/) platform. The methods rely on the Open Data Cube's [Statistician](https://github.com/opendatacube/odc-tools/tree/develop/libs/stats) library for orchestrating the machine learning predicitions on AWS's cloud infrastructure. 

## How to build and install the code

In the folder `dea_ml/`, run the following shell command:

```bash
pip install --extra-index-url="https://packages.dea.ga.gov.au" dea-ml

```

## Local testing of production analysis code

The ODC-statistician plugin that does the core analysis can be tested using the notebook [1_test_plugin.ipynb](1_test_plugin.ipynb).

* The ODC-stats plugin is called [PredGMS2](dea_ml/dea_ml/plugins/gm_ml_pred.py)
* The two primary analysis functions that this plugin references are in the [feature_layer](dea_ml/dea_ml/core/feature_layer.py) and [post_processing](dea_ml/dea_ml/core/post_processing.py) scripts.
* Two yamls are required to configure the code, one controls some of the inputs to the ML code, e.g. [ml_config_western](dea_ml/dea_ml/config/ml_config_western.yaml), and another controls some of the product specifications, e.g. [plugin_product_western](dea_ml/dea_ml/config/plugin_product_western.yaml).

## Running production code
 
 > Note, the following contains an example of running the code on the production EKS, the workflow should first be tested in DEV-EKS.
 
        DEV Cluster: `deafrica-dev-eks`
        PROD Cluster: `deafrica-prod-af-eks`
 
The steps to create a large scale cropland extent map using K8s and the ML-methods described in this repo are as follows:

1. Ensure the `ml_config` and `plugin_product` yamls are correct

2. Ensure the [Dockerfile](../Dockerfile) contains all the right files and libraries. If you alter the `Dockerfile` or the `dea_ml` code base you need to rebuild the image, this can triggered by changing the version number [here](../docker/version.txt) and creating a pull request.

3. Ensure the `datakube-apps` and `datakube` repositories have correctly configured pod/job templates. Make sure the image version and config urls are correct.  For production, the files to consider are:

    * Batch run job template: [06_stats_crop_mask.yaml](https://bitbucket.org/geoscienceaustralia/datakube-apps/src/master/workspaces/deafrica-prod-af/processing/statistician/06_stats_crop_mask.yaml)

    * Node group configuration (usually you shouldn't need to alter this): [workers.tf](https://bitbucket.org/geoscienceaustralia/datakube/src/37fbf47358d287aecefbe4f079bf5048f0295b82/workspaces/deafrica-prod-af/01_odc_eks/workers.tf#lines-126)  
    
    * Pod template (use this for testing): [06_stats_crop_mask_dev_pod.yaml](https://bitbucket.org/geoscienceaustralia/datakube-apps/src/master/workspaces/deafrica-prod-af/processing/statistician/06_stats_crop_mask_dev_pod.yaml)

4. Use your dev-box to access the `deafrica-prod-af-eks` environment

        setup_aws_vault deafrica-prod-af-eks
        ap deafrica-prod-af-eks

5. Create a dev-pod by running:
            
        kubectl apply -f workspaces/deafrica-prod-af/processing/statistician/06_stats_crop_mask_dev_pod.yaml
    

6. Access the dev-pod by runnning:

        kubectl exec -it crop-mask-dev-pod -n processing -- bash
        

7. Once in the dev-pod we need to create a large set of database 'tasks' that odc-stats will use as inputs to our ML pipeline (tile information). Run:

        odc-stats save-tasks --frequency annual --grid africa-10 gm_s2_semiannual
        

8. The database files created by running the above command now needs to be synced to the product's S3 bucket. Run:
    
        aws s3 sync . s3://deafrica-services/<product>/<version>/db-files --acl bucket-owner-full-control


9. To execute a batch run, we need to publish a list of tiles to AWS's Simple Queue Service. The command `cm-tsk` will use a geojson (e.g. `Western.geojson`) to clip the tasks to just a single region of Africa (defined by the extent of the geojson), and send those tasks/messages to SQS.

        cm-tsk --task-csv=gm_s2_semiannual_all.csv --geojson=/western/Western.geojson --outfile=/tmp/aez.csv --sqs deafrica-prod-af-eks-stats-crop-mask-eastern


9. Exit the dev-pod using `exit`, and then trigger the batch run using the command:

        kubectl -n processing apply -f workspaces/deafrica-prod-af/processing/06_stats_crop_mask.yaml


10. To monitor the batch run you can use https://mgmt.digitalearth.africa/?orgId=1


11. To move deadletter items back into the SQS queue, go into the dev pod, start python and run the following:
        
        >>> from odc.aws.queue import redrive_queue
        >>> redrive_queue('deafrica-prod-af-eks-crop-mask-eastern-deadletter', 'deafrica-dev-eks-stats-crop-mask-eastern')

---
## Other useful run notes

* Restarting the job if timeout errors prevent all messages being consumed:

     - Check the logs of multiple jobs
     - If multiple logs show the time-out error, then delete the job with `kubectl -n processing delete jobs crop-mask-ml-job`
     - Restart the job: `kubectl apply -f workspaces/deafrica-prod-af/processing/06_stats_crop_mask.yaml -n processing`

* To delete all messages from SQS queue:

     - go to AWS central app, open SQS, click on the queue you want to remove and hit the delete button

* To list tiles in a s3 bucket; useful to know if results have been successfully written to disk
        
        aws s3 ls s3:/deafrica-data-dev-af/crop_mask_western/
        
* To sync (copy) results in a s3 bucket to your local machine
        
        aws s3 sync s3:/deafrica-data-dev-af/crop_mask_western/ crop_mask_western

* If doing test runs, and you wish delete test geotifs from the dev bucket
        
        aws s3 rm --recursive s3:/deafrica-data-dev-af/folder --dryrun

* To test running one or two tiles in the dev-pod, you can directly run the `cm-pred` command

```
cm-pred run s3://deafrica-services/crop_mask_eastern/1-0-0/gm_s2_semiannual_all.db --config=${CFG} --plugin-config=${PCFG} --resolution=10 --threads=15 --memory-limit=100Gi --location=s3://deafrica-data-dev-af/{product}/{version} 719:721
```

* Useful kubectl commands you'll need

        kubectl get pods -n processing # See what pods are running
        kubectl get jobs -n processing # See what jobs are running
        kubectl logs <job-id> -n processing # Check the logs of a job
        kubectl delete jobs <job name> - processing # Delete a batch job
        kubectl delete pods <pod name> -n processing # Shut down pod
        kubectl get deployment -n processing
        kubectl -n processing describe pod crop-mask-dev-pod
        
---
## Additional information

**License:** The code in this notebook is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
Digital Earth Africa data is licensed under the [Creative Commons by Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) license.

**Contact:** If you need assistance, please post a question on the [Open Data Cube Slack channel](http://slack.opendatacube.org/) or on the [GIS Stack Exchange](https://gis.stackexchange.com/questions/ask?tags=open-data-cube) using the `open-data-cube` tag (you can view previously asked questions [here](https://gis.stackexchange.com/questions/tagged/open-data-cube)).
If you would like to report an issue with this notebook, you can file one on [Github](https://github.com/digitalearthafrica/crop-mask/issues).
