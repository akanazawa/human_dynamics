# Directions for Evaluation

## Test tfrecords

The evaluation scripts load the image sequences and ground truth labels from
the test tfrecords. The test tfrecords differ from the training tfrecords since
they do not have data augmentation and are not sharded. Each tfrecord
corresponds to one sequence. Each example in the tfrecord corresponds to the
tracklet for one person. All datasets except 3DPW only have one person.

For 3DPW and UPenn, follow the directions in [parepare_datasets.sh](prepare_datasets.sh) to
generate the test tfrecords. Also see [datasets.md](/doc/datasets.md) for their format. The directory with the test tfrecords should
follow the directory structure: `<tfrecords_dir>/<dataset_name>/<split>/*tfrecords`.


## Evaluating tfrecords

To call the evaluation script, run:
```bash
./src/evaluation/autorestart.py python -m src.evaluation.eval --tf_dir path_to_tfrecords --test_datasets tfrecords_test \
    --test_datasets 3dpw,penn_action,h36m --split val --load_path path_to_model_ckpt
```

Running this on the public model `hmmr_model.ckpt-1119816` outputs:

```
           Data      accel         kp      kp_pa     kp_pck     joints  joints_pa mesh_posed mesh_tpose
           3dpw    0.01532    5.90772    5.48809    0.92961    0.11688    0.07266    0.13934    0.02680
           h36m    0.00882    4.93181    4.00981    0.95606    0.08371    0.05692   -1.00000   -1.00000
    penn_action    0.02796    8.87832    8.51331    0.79553   -1.00000   -1.00000   -1.00000   -1.00000
```
