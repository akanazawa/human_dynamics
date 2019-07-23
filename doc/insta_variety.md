# InstaVariety Dataset

![Teaser Gif](../resources/instavariety.gif)

## Raw Data
In addition to releasing our pre-processed tfrecords (see below) that contain the exact InstaVariety data we introduced in [Learning 3D Human Dynamics from Video](https://akanazawa.github.io/human_dynamics/ "Appearing in CVPR 19 proceedings"), we also release the full dataset that we downloaded from Instagram. In particular, to generate InstaVariety, we scraped 92 movement-oriented Instagram tags ([full tag list](../datasets/instavariety/tag_list.txt)) for 1000 posts each. Instagram posts can be either videos or photos, so after filtering out the photos, we end up with many fewer videos than 1000 per tag. Across the 92 tags, we collected 28,272 videos in total. We provide a file [`InstaVariety.json`](https://drive.google.com/file/d/11Xo_2JsoL2S4n8pcofnX_j76Xb7H6OL6/view?usp=sharing) that contains the links to each of these videos, along with relevant metadata for each video, including:

```javascript
{
 'edge_media_preview_like': {'count': int}, 
 'urls': [str], //we use as filename
 'edge_media_to_caption': {'edges': [{'node': {'text': str}}]}, //caption str also contains tags
 'dimensions': {'width': int, 'height': int}, 
 'tags': [str], //list of tag strings, without '#'
 'edge_media_to_comment': {'count': int}, 
 'video_view_count': int, 
 'comments_disabled': boolean, //(note: not always available)
 'download_tag': str, //corresponds to tag we used to identify and download this video
 'edge_liked_by': {'count': int}, 
 'shortcode': str, //the shortcode is used to construct the download link
 'taken_at_timestamp': int, //reported in unix time
 'video_link': str, //of the form 'https://www.instagram.com/p/{}'.format(shortcode)
 'is_video': boolean, //should always be True
 'id': str
}
```

We also provide a convenience script [`download_insta_variety.py`](../datasets/instavariety/download_insta_variety.py) for downloading the videos. The script relies on `youtube-dl`, which can be downloaded from [here](https://ytdl-org.github.io/youtube-dl/download.html). The script can be run as:

```console
foo@bar:~$ python download_insta_variety.py --savedir /path/to/your/save/directory
```

To use the same test split of videos that we used in [Learning 3D Human Dynamics from Video](https://akanazawa.github.io/human_dynamics/ "Appearing in CVPR 19 proceedings"), we provide the list of train and test split videos in [`insta_variety_train.txt`](../datasets/instavariety/insta_variety_train.txt) and [`insta_variety_test.txt`](../datasets/instavariety/insta_variety_test.txt), respectively. Within each `.txt` file, the videos are listed according to `download_tag/video_id`.

## Pre-processed tfrecords
We provide our pre-processed tfrecords, which can be accessed [here](https://drive.google.com/file/d/1C20Wdxs5VL4IC7Hf6UieWrI8aAzOd94B/view?usp=sharing). Details of the tfrecord format can be found in [doc/datasets.md](datasets.md).

### Visualizing the tfrecords
Once you download the tfrecords, you can visualize them by:
```console
foo@bar:~$ python -m src.datasets.visualize_train_tfrecords --data_rootdir /path/to/your/tfrecord/rootdir --dataset insta_variety
```
Where `/path/to/your/tfrecord/rootdir` contains the `insta_variety` directory with tfrecords in a `train` subdirectory.

## Generating tfrecords
To generate the tfrecords yourself, we provide the trajectories we derived from linking per-frame OpenPose detections for the videos we used in training and evaluation, which can be accessed [here](https://drive.google.com/file/d/1i_p_uurnGSMynrGzKoiY9tL_pn0IdONR/view?usp=sharing). The directory structure is:
```bash
InstaVariety_tracks
├── instagram tag id
    ├── instagram video id
        ├── track id
            ├── per-frame JSON files with OpenPose keypoints
```
The format for each per-frame JSON file with OpenPose keypoints is
```javascript
{
 'frame': int, //frame number
 'imloc': str, //image name. note that ffmpeg extraction is 1-indexed, while the frame number is 0-indexed
 'track_id': //this may not be unique. we split tracks based on shot detection in the video.
 'bbox': {'xmin': float, 'ymin': float, 'xmax': float, 'ymax': float},
 'keypoint name': { //these are things like 'R Knee','R Big Toe', 'Hip', 'L Ankle', etc.
   'x': float,
   'y': float,
   'logits': float,
   'probability': null
 }
}
```
The commands for generating tfrecords using these tracks can be found in [prepare_datasets.sh](../prepare_datasets.sh).

To generate train tfrecords:
```
python -m src.datasets.video_in_the_wild_to_tfrecords --data_directory ${INSTA_TRACKS_DIR} --output_directory ${OUT_DIR}/insta_variety --num_copy ${INSTA_NUM_COPIES} --pretrained_model_path ${HMR_MODEL} --image_directory ${INSTA_FRAMES_DIR} --video_list ${INSTA_VIDEO_LIST} --split train
```

To generate test tfrecords:
```
python -m src.datasets.video_in_the_wild_to_tfrecords --data_directory ${INSTA_TRACKS_DIR} --output_directory ${OUT_DIR}/insta_variety --num_copy 1 --pretrained_model_path ${HMR_MODEL} --image_directory ${INSTA_FRAMES_DIR} --video_list ${INSTA_VIDEO_LIST} --split test
```

### Citation
If you use this data for your research, please cite:
```
@InProceedings{humanMotionKZFM19,
  title={Learning 3D Human Dynamics from Video},
  author = {Angjoo Kanazawa and Jason Y. Zhang and Panna Felsen and Jitendra Malik},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
```
