# General prediction pipeline and data indexing walkthrough

## Short Summary

1. Get into Sandbox notebooks, in terminal, use shell command to create the target data folder,
  ````bash
  mkdir -p /g/data/<you/sub/folder>
  ````

2. If there were no indexed 6 month geomedian data, we must create them. Refer to Step-2

    - odc-stats save-task with time range and tile info

    - odc-stats run to collect all band tifs for the tiles

3. After we collected the tifs in the local folder, we can merge those tifs into xarray ds.
   Refer to the psuedo code sample notebok.

  - Merge tifs into datasets then generate predictions.

  - Save the prediction results, json and mask tif,

4. Docker run postgresql image. Refer to Step-2 about indexing postgresql

  - Setup config ```.datacube.conf``` or environment variables
  - create db datacube
  - ```datacube -v system init``` to initialize the metadata_types and other table schemas
  - ```datacube product add <product>.yaml```

2. AWS S3 data sync from Sandbox or devbox onto S3  buckets with the same prefix as defined in the stac json generated

  - Sort out the s3 access, make sure the ```s3://deafrica-data-dev-af``` is writable
    and sufficient control to allow ```--acl public-read```

  - ```aws s3 sync <local data> <remote s3 bucket and prefix> --acl public-read```

## Notes

Refer the details in each steps described in the same folder.
