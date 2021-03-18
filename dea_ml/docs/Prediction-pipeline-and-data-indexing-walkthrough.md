# Prediction pipeline and data indexing walkthrough

## Summary

1. Sandbox generate the prediction results, json and mask tifs

- odc-stats save-task with time range and tile info

- odc-stats run to collect all band tifs for the tiles

- Merge tifs into daasets then generate predictions and stac json file.

2. Docker run postgresql image

- Setup config ```.datacube.conf``` or environment variables
- create db datacube
- ```datacube -v system init``` to initialize the metadata_types and other table schemas
- ```datacube product add <product>.yaml```

3. AWS S3 data sync from Sandbox or devbox onto S3  buckets with the same prefix as defined in the stac json generated

- Sort out the s3 access, make sure the ```s3://deafrica-data-dev-af``` is writable
 and sufficient control to allow ```--acl public-read```
- ```aws s3 sync <local data> <remote s3 bucket and prefix> --acl public-read```

## Notes

Refer the details in each steps described in other wiki pages.
