There should be 5 results.
## Direct LSTM

`python 1_Clusterinfo_based_data_shard_slice.py`

`python 2_1_Direct_training_process.py`

`python 2_2_Direct_testing_processing.py`

## SISA LSTM

### Random Shard: weighted mean | weighted error

You should run the following scripts for $idx from 1 to num_shards:

`python 3_1_SISA_Random_Sort_Shard_Sub_model_training.py --shard_index $idx`

`python 3_2_SISA_Random_Sort_Shard_Loss_weight_training.py  --shard_index $idx`

Then, run the following script:

`python 3_3_SISA_Random_Sort_Aggerate_model_weg_avg.py`

### Cluster-assign Shard: weighted mean | weighted error

You should run the following scripts for $idx from 1 to num_shards:

`python 4_1_SISA_Clustering_Sort_Shard_Sub_model_training.py --shard_index $idx`

`python 4_2_SISA_Clustering_Sort_Shard_Loss_weight_training.py --shard_index $idx`

Then, run the following script:

`python 4_3_SISA_Clustering_Sort_Aggerate_model_weg_avg.py`

