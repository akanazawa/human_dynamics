# Learning 3D Human Dynamics from Video

Angjoo Kanazawa*, Jason Zhang*, Panna Felsen*, Jitendra Malik

University of California, Berkeley
(* Equal contribution)

[Project Page](https://akanazawa.github.io/human_dynamics/)

![Teaser Image](resources/overview.jpg)

### Requirements
- Python 3 (tested on version 3.5)
- [TensorFlow](https://www.tensorflow.org/) (tested on version 1.8)
- [PyTorch](https://pytorch.org/) for AlphaPose, PoseFlow, and NMR (tested on
  version 0.4.0)
- [AlphaPose/PoseFlow](https://github.com/akanazawa/AlphaPose)
- [Neural Mesh Renderer](https://github.com/daniilidis-group/neural_renderer)
  for rendering results. See below.
- [CUDA](https://developer.nvidia.com/cuda-downloads) (tested on CUDA 9.0 with Titan 1080 TI)
- ffmpeg (tested on version 3.4.4)

There is currently no CPU-only support.

### License
Please note that while our code is under BSD, the SMPL model and datasets we use have their own licenses that must be followed.

### Contributions
- Windows build and Unity port. Thanks George @ZjuSxh! https://github.com/Zju-George/human_dynamics

### Installation

#### Setup virtualenv
```
virtualenv venv_hmmr -p python3
source venv_hmmr/bin/activate
pip install -U pip
pip install numpy  # Some of the required packages need numpy to already be installed.
deactivate
source venv_hmmr/bin/activate
pip install -r requirements.txt
```


#### Install External Dependencies.
Neural Mesh Renderer and AlphaPose for rendering results:
```
cd src/external
sh install_external.sh
```

The above script also clones my fork of [AlphaPose/PoseFlow](https://github.com/akanazawa/AlphaPose),
which is necessary to run the demo to extract tracks of people in videos. Please
follow the directions in [the installation](https://github.com/akanazawa/AlphaPose/tree/pytorch#installation),
in particular running `pip install -r requirements.txt` from
`src/external/AlphaPose` and downloading the trained models.

If you have a pre-installed version of AlphaPose, symlink the directory in
`src/external`. 
The only change that my fork has is a very minor modification in
AlphaPose/pytorch branch's `demo.py`: see [this commit](https://github.com/akanazawa/AlphaPose/commit/ed9cd3c458f1e61145c1b10f87bd37cba53233cd),
copy over the changes in `demo.py`. 


### Demo

1. Download the pre-trained models. Place the `models` folder as a top-level
directory.

```
wget http://angjookanazawa.com/cachedir/hmmr/hmmr_models.tar.gz && tar -xf hmmr_models.tar.gz
```
2. Download the `demo_data` videos. Place the `demo_data` folder as a top-level
directory.

```
wget http://angjookanazawa.com/cachedir/hmmr/hmmr_demo_data.tar.gz && tar -xf hmmr_demo_data.tar.gz
```

3. Run the demo. This code runs AlphaPose/PoseFlow for you.
Please make sure AlphaPose can be run on a directory of images if you are having 
any issues. 

Sample usage:

```
# Run on a single video:
python -m demo_video --vid_path demo_data/penn_action-2278.mp4 --load_path models/hmmr_model.ckpt-1119816

# If there are multiple people in the video, you can also pass a track index:
python -m demo_video --track_id 1 --vid_path demo_data/insta_variety-tabletennis_43078913_895055920883203_6720141320083472384_n_short.mp4 --load_path models/hmmr_model.ckpt-1119816

# Run on an entire directory of videos:
python -m demo_video --vid_dir demo_data/
```

This will make a directory `demo_output/<video_name>`, where intermediate
tracking results and our results are saved as video, as well as a pkl file. 
Alternatively you can specify the output directory as well. See `demo_video.py`


### Training code

See [doc/train](doc/train.md).

### Data

#### InstaVariety

![Insta-Variety Teaser](resources/instavariety.gif)


We provided the raw list of videos used for InstaVariety, as well as the
pre-processed files in tfrecords. Please see
[doc/insta_variety.md](doc/insta_variety.md) for more details..

### Citation
If you use this code for your research, please consider citing:
```
@InProceedings{humanMotionKZFM19,
  title={Learning 3D Human Dynamics from Video},
  author = {Angjoo Kanazawa and Jason Y. Zhang and Panna Felsen and Jitendra Malik},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
