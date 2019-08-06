## Pre-reqs

### Download required models

1. (Same as demo) Download the pre-trained models. Place the `models` folder as a top-level
directory.

```
wget http://angjookanazawa.com/cachedir/hmmr/hmmr_models.tar.gz && tar -xf hmmr_models.tar.gz
```

### Download datasets.
Download these datasets:

- [Penn Action Dataset](https://dreamdragon.github.io/PennAction/)
- [Human3.6M](http://vision.imar.ro/human3.6m/description.php)
- [Mosh Data on CMU and JointLimits in HMR](https://drive.google.com/file/d/1b51RMzi_5DIHeYh2KNpgEs8LVaplZSRP/view?usp=sharing) Please note that the usage of this data is for [**non-comercial scientific research only**](http://mosh.is.tue.mpg.de/data_license).

If you use any of the data above, please cite the corresponding datasets and
follow their licenses:
```
article{Loper:SIGASIA:2014,
  title = {{MoSh}: Motion and Shape Capture from Sparse Markers},
  author = {Loper, Matthew M. and Mahmood, Naureen and Black, Michael J.},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  volume = {33},
  number = {6},
  pages = {220:1--220:13},
  publisher = {ACM},
  address = {New York, NY, USA},
  month = nov,
  year = {2014},
  url = {http://doi.acm.org/10.1145/2661229.2661273},
  month_numeric = {11}
}
```
- [Insta-Variety](insta_variety.md)
- [Vlog-people](vlog_people.md)
For evaluation, download the 3DPW dataset:
- [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

## TFRecord Generation

All the data has to be converted into TFRecords and saved to a `DATA_DIR` of
your choice.

1. Make `DATA_DIR` where you will save the tf_records. For ex:
```
mkdir ~/hmmr/tf_datasets/
```

2. Edit `prepare_datasets.sh`, with paths to where you downloaded the datasets,
and set `DATA_DIR` to the path to the directory you just made. Please read the
comments in `prepare_datasets.sh` for each dataset. Some require running a
pre-processing script, which is spelled out in `prepare_datasets.sh`

3. Follow the direction in `prepare_datasets.sh`, for each dataset, uncomment
each command and from the root HMMR directly (where README is), run `prepare_datasets.sh`:
```
sh prepare_datasets.sh
```

This takes a while! Start with UPenn and Human3.6M. Insta_variety and Vlog
people are optional.

5. In `do_train.sh` and/or `src/config.py`, set `DATA_DIR` to the path where you saved the
tf_records.


## Training
Now we can start training.
A sample training script (with parameters used in the paper) is in
`do_train.sh`.

Update the path to  in the beginning of this script and run:
```
sh do_train.sh
```

The training write to a log directory that you can specify.
Setup tensorboard to this directory to monitor the training progress like so:
![Teaser Image](https://akanazawa.github.io/human_dynamics/resources/images/tboard_ex.png)

It's important to visually monitor the training! Make sure that the images
loaded look right.

### config
`do_train.sh` will use the default weights/setting. But see
[src/config.py](/src/config.py) for more options.


## Note on data.
Unfortunately, due to the licensing, the Human3.6M Mosh data is no longer
available.
This means you will not be able to reproduce exactly the public model since the
supervision will not have the mosh values. Human3.6M data produced above will
still have the 3D joint annotation, but not the SMPL values. 
We have re-trained our model on this setting where Human3.6M mosh data is not available, see [doc/eval.md](/doc/eval.md#evaluating-tfrecords)
