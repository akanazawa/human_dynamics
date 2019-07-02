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
./src/evaluation/autorestart.py python -m src.evaluation.eval --tf_dir path_to_tfrecords --split val
  --test_datasets tfrecords_test --test_datasets 3dpw,penn_action,h36m --load_path path_to_model_ckpt
```

Running the evaluation on the test set using the public model `hmmr_model.ckpt-1119816` should give you:

```
       Data      accel         kp      kp_pa     kp_pck     joints  joints_pa mesh_posed mesh_tpose
       3dpw    0.01532    5.90772    5.48809    0.92961    0.11688    0.07266    0.13934    0.02680
       h36m    0.00882    4.93181    4.00981    0.95606    0.08371    0.05692   -1.00000   -1.00000
penn_action    0.02796    8.87832    8.51331    0.79553   -1.00000   -1.00000   -1.00000   -1.00000
```

We also trained a model with H3.6M (without Mosh data), Penn Action, and InstaVariety. Here are the numbers we get for that model:

```
       Data      accel         kp      kp_pa     kp_pck     joints  joints_pa mesh_posed mesh_tpose
       3dpw    0.01635    6.20286    5.79252    0.91890    0.12803    0.07731    0.14852    0.02756
       h36m    0.00851    4.85473    3.98626    0.95704    0.08895    0.05939   -1.00000   -1.00000
penn_action    0.02894    9.24552    8.71280    0.78622   -1.00000   -1.00000   -1.00000   -1.00000
```



Error metrics:
* `accel`: Acceleration error (mm/s^2)
* `kp`: 2D KP Error (pixels)
* `kp_pa`: 2D KP Error after optimal scale and translation (pixels)
* `kp_pck`: Percentage of Correct Keypoints (%)
* `joints`: MPJPE (mm)
* `joints_pa`: MPJPE after Procrustes Alignment (mm)
* `mesh_posed`: Mesh error posed (mm)
* `mesh_tposed`: Mesh error unposed (mm)
