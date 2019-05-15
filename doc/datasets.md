# TFRecord format

We provide InstaVariety and VLogPeople in a precomputed tfrecord format. We also
provide scripts for saving other datsets in this format.
Below are the descriptions of this common format.

## Train
Train tfrecords contain key value pairs:

```
# Meta data
'meta/N': Number of frames in this video (int).
'image/filenames': list of file names (bytes).
'image/heightwidths': (2,) image shape (int).
'image/centers': (N*2,) center of the bounding box (after jitter) used to
    preprocess each frame (int).
'image/scale_factors': (N*2,) scale factor (after jitter) used to preprocess
    each frame (float).
'image/crop_pts': (N*2,) start point for the crop for each frame (float).

# Annotations
'image/xys': (N*14*2,) xy location of the first 14 common joints
    ['R Heel', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Heel', 'R Wrist',
     'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck',
     'Head'] (float).
'image/visibilities': (N*14,) visibility on those 14 (int).
'image/face_pts': (N*5*3,): (x y,vis) tuple for 5 face points
    ['Nose', 'L Eye', 'R Eye', 'L Ear', 'R Ear'] (float).
'image/toes_pts': (N*6*3,): (x y,vis) tuple for the new 6 toe points
    ['L Big Toe', 'R Big Toe', 'L Small Toe', 'R Small Toe', 'L Ankle',
     'R Ankle'] (float).
# Image features
'image/encoded: byte array of JPEG encoding of each RGB frame.
'image/phi: byte array of 2048D precomputed HMR features for each frame.
```

Note the tfrecord also contains the keys related to 3D groundtruth but all the
datasets we provide have no ground truth so the values for these keys are all
just 0.
```
'mosh/shape'
'mosh/gt3ds'
'mosh/poses'
'image/cams'
'meta/has_3d'
'meta/has_3d_joints'
```

### Visualize
These tfrecords can be visualized with

```console
foo@bar:~$ python -m src.datasets.visualize_train_tfrecords --data_rootdir /path/to/your/tfrecord/rootdir --dataset your_dataset_directory
```
Where `/path/to/your/tfrecord/rootdir` contains the `your_dataset_directory` directory with tfrecords in a `train` subdirectory.

## Test

In addition to the above fields, the test tfrecords have:

```
'meta/time_pts': (2,) start and end frame of the original video that the track
    corresponds to (int).
```

