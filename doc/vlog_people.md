# VLOG-people Dataset

![Teaser Gif](../resources/vlog.gif)

The authors of [From lifestyle vlogs to everyday interactions](https://web.eecs.umich.edu/~fouhey/2017/VLOG/) generously shared a people-split of their VLOG dataset, where OpenPose was found to fire well, from which we used a subset to form our VLOG-people dataset. The video ids that we used can be found in [datasets/vlog/vlog_ids.txt](../datasets/vlog/vlog_ids.txt).

## Pre-processed tfrecords
We provide our pre-processed tfrecords, which can be accessed [here](https://drive.google.com/file/d/1AgtQ26ENxeorfsYYKp6mL8CodJJXbDCH/view?usp=sharing). Details of the tfrecord format can be found in [doc/datasets.md](datasets.md).

### Citation
If you use this data for your research, please cite:
```
@inproceedings{fouhey2018lifestyle,
  title={From lifestyle vlogs to everyday interactions},
  author={Fouhey, David F and Kuo, Wei-cheng and Efros, Alexei A and Malik, Jitendra},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4991--5000},
  year={2018}
}
```
```
@InProceedings{humanMotionKZFM19,
  title={Learning 3D Human Dynamics from Video},
  author = {Angjoo Kanazawa and Jason Y. Zhang and Panna Felsen and Jitendra Malik},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
```
