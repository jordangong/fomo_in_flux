## Downloading DataComp subset used in our experiments

### Download pool
For downloading the pool shards, run:
```bash
python download_pool_shards.py --scale small --data_dir /path/to/store/datacomp-subset --metadata_dir /path/to/store/datacomp-subset/metadata
```

For example, on the cambridge cluster, I run:
```bash
python download_pool_shards.py --scale small --data_dir /home/vu214/rds/rds-shared-data-HM7VddDwcug/shared-datasets/foundation_adaptation_datasets/continualfomo_224_singled/datasets/datacomp1b --metadata_dir /home/vu214/rds/rds-shared-data-HM7VddDwcug/shared-datasets/foundation_adaptation_datasets/continualfomo_224_singled/datasets/datacomp1b/metadata
```

### Download statistics
Our downloaded subset (download spawned on 04/05/24 and took around 18h18mins to complete) contains `1,251,473` image-text pairs in total. 
