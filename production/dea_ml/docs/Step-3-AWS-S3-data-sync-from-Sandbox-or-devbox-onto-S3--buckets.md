# prepare data on S3

We can use ```awscli``` to sync seed data into s3 to validate the s3 path as well as other setups.
The s3 url link of the stac json was also included in the stac json it self, like below

```json
"s3://deafrica-data-dev-af/crop_mask_eastern/v0.1.3/x+048/y+010/2019/crop_mask_eastern_x+048_y+010_2019.json"
```

The linke parts include

1. bucket name
2. product name
3. product version
4. ```<x>/<y>``` tile index
5. time range, ```2019``` here is for the whole year. Otherwise, put something like, ```2019-01--P12M```, which was defined
  in the ```odc.stats.model.DatatimeRange```.

It is possible to use localstack to mock s3. But it needs extra tweaking.

Ideally, we do the mock s3 as below,

1. create the local mirror bucket

  ```bash
  aws s3 mb s3://deafrica-data-dev-af --endpoint-url=htp://192.168.0.19:4566
  ```

2. sync data onto the bucket


```bash
  aws s3 sync src/seed/v0.1.4 s3://deafrica-data-dev-af/crop_mask_eastern/v0.1.4 --endpoint-url=htp://192.168.0.19:4566 \
   --acl public-read
```

### Notes

Remember to add ``` --acl public-read``` when sync the data to real s3. Prepare the data in ```v0.1.4``` with the
exact folder structure as the s3 key prefix you want. Then, sync data,

  ```bash
  aws s3 sync v0.1.4/ s3://deafrica-data-dev-af/crop_mask_eastern/v0.1.4 --acl public-read
  ```

which just following the [aws s3 sync command](https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html).

  ```bash
  aws s3 sync <local data path> <remote s3 bucket and prefix> --acl public-read
  ```

Remember to delete the data recursively when you finished the testing.

  ```bash
  aws s3 rm --recursive s3://deafrica-data-dev-af/crop_mask_eastern
  ```
