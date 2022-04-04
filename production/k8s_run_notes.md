# Running the cropmask using kubernetes

> These are notes for running a crop-mask job manually through the command line after logging into the dev or prod k8s cluster (using a devbox). The notes here are largely superseeded by the Argo run procedure explained in the `README` file, but these notes are preserved here for posterity.

* The ODC-stats plugin is called [PredGMS2](cm_tools/cm_tools/gm_ml_pred.py)
* The two primary functions that this plugin references are in the [feature_layer](cm_tools/cm_tools/feature_layer.py) and [post_processing](cm_tools/cm_tools/post_processing.py) scripts.
* A yaml is required to configure the plugin, e.g. [config_western](cm_tools/cm_tools/config/config_western.yaml)

## Running production code

 > Note, the following contains an example of running the code on the production EKS, the workflow should first be tested in DEV-EKS.

    DEV Cluster: deafrica-dev-eks
    PROD Cluster: deafrica-prod-af-eks

The steps to create a large scale cropland extent map using K8s and the ML-methods described in this repo are as follows:

1. Ensure the `config_<region>` yaml is correct for the region of Africa you are running. i.e. one of these files [here](https://github.com/digitalearthafrica/crop-mask/tree/main/production/cm_tools/cm_tools/config)

2. Ensure the [Dockerfile](../Dockerfile) contains all the right files and libraries. If you alter the `Dockerfile` or the `cm_tools` code base you need to rebuild the image, this can triggered by changing the version number [here](../docker/version.txt) and creating a pull request.

3. Ensure the `datakube-apps` and `datakube` repositories have correctly configured pod/job templates. Make sure the image version and config urls are correct.  For production, the files to consider are:
    * Batch run job template: [06_stats_crop_mask.yaml](https://bitbucket.org/geoscienceaustralia/datakube-apps/src/master/workspaces/deafrica-prod-af/processing/statistician/06_stats_crop_mask.yaml)
      * Node group configuration (usually you shouldn't need to alter this): [workers.tf](https://bitbucket.org/geoscienceaustralia/datakube/src/37fbf47358d287aecefbe4f079bf5048f0295b82/workspaces/deafrica-prod-af/01_odc_eks/workers.tf#lines-126)  
      * Pod template (use this for testing): [06_stats_crop_mask_dev_pod.yaml](https://bitbucket.org/geoscienceaustralia/datakube-apps/src/master/workspaces/deafrica-prod-af/processing/statistician/06_stats_crop_mask_dev_pod.yaml)

4. Use your dev-box to access the `deafrica-prod-af-eks` environment:
        setup_aws_vault deafrica-prod-af-eks
        ap deafrica-prod-af-eks

5. Create a dev-pod by running:
        kubectl apply -f workspaces/deafrica-prod-af/processing/statistician/06_stats_crop_mask_dev_pod.yaml

6. Access the dev-pod by runnning:
        kubectl exec -it crop-mask-dev-pod -n processing -- bash

7. Once in the dev-pod we need to create a large set of database 'tasks' that odc-stats will use as inputs to our ML pipeline (tile information). Run:
        odc-stats save-tasks --frequency annual --grid africa-10 gm_s2_semiannual

8. The database files created by running the above command now needs to be synced to the product's S3 bucket. Run:
        aws s3 sync . 's3://deafrica-services/<product>/<version>/db-files' --acl bucket-owner-full-control

9. To execute a batch run, we need to publish a list of tiles to AWS's Simple Queue Service. The command `cm-tsk` will use a geojson (e.g. `Western.geojson`) to clip the tasks to just a single region of Africa (defined by the extent of the geojson), and send those tasks/messages to SQS.
        cm-task --task-csv=gm_s2_semiannual_all.csv --geojson=/western/Western.geojson --outfile=/tmp/aez.csv --sqs deafrica-prod-af-eks-stats-crop-mask --db=s3://deafrica-services/crop_mask_eastern/1-0-0/gm_s2_semiannual_all.db

10. Exit the dev-pod using `exit`, and then trigger the batch run using the command:
        kubectl -n processing apply -f workspaces/deafrica-prod-af/processing/statistician/06_stats_crop_mask.yaml

11. To monitor the batch run you can use:
    * CPU and memory monitoring: `https://mgmt.digitalearth.africa/d/wIVvTqR7k/crop-mask`
    * SQS and instance monitoring: `https://monitor.cloud.ga.gov.au/d/n2TdQCnnz/crop-mask-dev-deafrica?orgId=1`

12. To move deadletter items back into the SQS queue, go into the dev pod, start python and run the following:

    `redrive-queue --from-queue <queue-name>`

## Other useful run notes

* Restarting the job if timeout errors prevent all messages being consumed:
  * Check the logs of multiple jobs:
  * If multiple logs show the time-out error, then delete the job with: `kubectl -n processing delete jobs crop-mask-ml-job`

  * Restart the job: `kubectl apply -f workspaces/deafrica-prod-af/processing/06_stats_crop_mask.yaml -n processing`

* To delete all messages from SQS queue:
  * go to AWS central app, open SQS, click on the queue you want to remove and hit the delete button

* To list tiles in a s3 bucket; useful to know if results have been successfully written to disk
  `aws s3 ls s3://deafrica-data-dev-af/crop_mask_western/`

* To sync (copy) results in a s3 bucket to your local machine
  `aws s3 sync s3://deafrica-data-dev-af/crop_mask_western/ crop_mask_western`

* If doing test runs, and you wish delete test geotifs from the dev bucket
  `aws s3 rm --recursive s3://deafrica-data-dev-af/crop_mask_western --dryrun`

* To test running one or two tiles in the dev-pod, you can directly run the `cm-pred` command

  odc-stats run s3://deafrica-services/crop_mask_eastern/1-0-0/gm_s2_semiannual_all.db --config=${CFG} --resolution=10 --threads=15 --memory-limit=100Gi --location=s3://deafrica-data-dev-af/{product}/{version} 719:721

* Useful kubectl commands you'll need
        # See what pods are running
        kubectl get pods -n processing

        # See what jobs are running
        kubectl get jobs -n processing
        
        # Check the logs of a job
        kubectl logs <job-id> -n processing 
        
        # Delete a batch job
        kubectl delete jobs <job name> - processing 
        
        # Shut down pod
        kubectl delete pods <pod name> -n processing 
        
        kubectl get deployment -n processing
        
        kubectl -n processing describe pod crop-mask-dev-pod 

## Alex's Notes

### Save Tasks

``` bash
odc-stats save-tasks \
  --frequency annual \
  --grid africa-10 gm_s2_semiannual \
  --tiles=215:218,75:78 \
  --year=2019
```

### Run

``` bash
odc-stats run gm_s2_semiannual_2021--P1Y.db \
  --config production/cm_tools/cm_tools/config/config_eastern.yaml \
  --resolution=10 \
  --threads=15 \
  --memory-limit=100Gi \
  --location=file:///tmp/cm_test/{product}/{version}
```

### Zip

`zip -r ~/cm_test.zip 2019--P1Y/`
