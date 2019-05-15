# InstaVariety Dataset

![Teaser Gif](../resources/instavariety.gif)

## Raw Data
In addition to releasing our pre-processed tfrecords (see below) that contain the exact InstaVariety data we introduced in [Learning 3D Human Dynamics from Video](https://akanazawa.github.io/human_dynamics/ "Appearing in CVPR 19 proceedings"), we also release the full dataset that we downloaded from Instagram. In particular, to generate InstaVariety, we scraped 92 movement-oriented Instagram tags ([full tag list](../datasets/instavariety/tag_list.txt)) for 1000 posts each. Instagram posts can be either videos or photos, so after filtering out the photos, we end up with many fewer videos than 1000 per tag. Across the 92 tags, we collected 28,272 videos in total. We provide a file [`InstaVariety.json`](https://drive.google.com/file/d/11Xo_2JsoL2S4n8pcofnX_j76Xb7H6OL6/view?usp=sharing) that contains the links to each of these videos, along with relevant metadata for each video, including:

```javascript
{
 'edge_media_preview_like': {'count': int}, 
 'urls': [str], //(note: we use as filename) 
 'edge_media_to_caption': {'edges': [{'node': {'text': str}}]}, //(note: caption str also contains tags) 
 'dimensions': {'width': int, 'height': int}, 
 'tags': [str], //(note: list of tag strings, without '#')
 'edge_media_to_comment': {'count': int}, 
 'video_view_count': int, 
 'comments_disabled': boolean, //(note: not always available)
 'download_tag': str, //(note: corresponds to tag we used to identify and download this video) 
 'edge_liked_by': {'count': int}, 
 'shortcode': str, //(note: the shortcode is used to construct the download link) 
 'taken_at_timestamp': int, //(note: reported in unix time) 
 'video_link': str, //(note: of the form 'https://www.instagram.com/p/{}'.format(shortcode))
 'is_video': boolean, //(note: should always be True) 
 'id': str
}
```

We also provide a convenience script [`download_insta_variety.py`](../datasets/instavariety/download_insta_variety.py) for downloading the videos. The script relies on `youtube-dl`, which can be downloaded from [here](https://ytdl-org.github.io/youtube-dl/download.html). The script can be run as:

```console
foo@bar:~$ python download_insta_variety.py --savedir /path/to/your/save/directory
```

To use the same test split of videos that we used in [Learning 3D Human Dynamics from Video](https://akanazawa.github.io/human_dynamics/ "Appearing in CVPR 19 proceedings"), we provide the list of test split videos in [`test_set_video_list.txt`](../datasets/instavariety/test_set_video_list.txt).

## Pre-processed tfrecords
We provide our pre-processed tfrecords, which can be accessed [here](https://drive.google.com/file/d/1C20Wdxs5VL4IC7Hf6UieWrI8aAzOd94B/view?usp=sharing). Details of the tfrecord format can be found in [doc/datasets.md](datasets.md).

### Visualizing the tfrecords
Once you download the tfrecords, you can visualize them by:
```console
foo@bar:~$ python -m src.datasets.visualize_train_tfrecords --data_rootdir /path/to/your/tfrecord/rootdir --dataset insta_variety
```
Where `/path/to/your/tfrecord/rootdir` contains the `insta_variety` directory with tfrecords in a `train` subdirectory.

### Citation
If you use this data for your research, please cite:
```
@InProceedings{humanMotionKZFM19,
  title={Learning 3D Human Dynamics from Video},
  author = {Angjoo Kanazawa and Jason Y. Zhang and Panna Felsen and Jitendra Malik},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
```
