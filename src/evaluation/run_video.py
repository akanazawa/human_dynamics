from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from tqdm import tqdm

from src.util.render.nmr_renderer import (
    VisRenderer,
    visualize_img,
    visualize_img_orig,
)
from src.util.common import resize_img
from src.util.smooth_bbox import get_smooth_bbox_params


IMG_SIZE = 224


def process_video(model, config, im_paths, kps, pred_dir, min_frame=0,
                  max_frame=None):
    bbox_params_smooth, s, e = get_smooth_bbox_params(kps, vis_thresh=0.1)
    min_f = max(s, min_frame)
    if max_frame:
        max_f = min(e, max_frame)
    else:
        max_f = min(e, len(kps))

    images = []
    images_orig = []
    for i in tqdm(range(min_f, max_f)):
        proc_params = process_image(
            im_path=im_paths[i],
            bbox_param=bbox_params_smooth[i],
        )
        images.append(proc_params.pop('image'))
        images_orig.append(proc_params)

    preds = model.predict_all_images(images)
    render_preds(
        output_path=pred_dir,
        config=config,
        preds=preds,
        images=images,
        images_orig=images_orig,
    )


def process_image(im_path, bbox_param):
    """
    Processes an image, producing 224x224 crop.

    Args:
        im_path (str).
        bbox_param (3,): [cx, cy, scale].
        visualize (bool).

    Returns:
        dict: image, im_shape, center, scale_factors, start_pt.
    """
    image = imread(im_path)
    center = bbox_param[:2]
    scale = bbox_param[2]

    # Pre-process image to [-1, 1]
    image = ((image / 255.) - 0.5) * 2
    image_scaled, scale_factors = resize_img(image, scale)
    center_scaled = np.round(center * scale_factors).astype(np.int)

    # Make sure there is enough space to crop 224x224.
    image_padded = np.pad(
        array=image_scaled,
        pad_width=((IMG_SIZE,), (IMG_SIZE,), (0,)),
        mode='edge'
    )
    height, width = image_padded.shape[:2]
    center_scaled += IMG_SIZE

    # Crop 224x224 around the center.
    margin = IMG_SIZE // 2

    start_pt = (center_scaled - margin).astype(int)
    end_pt = (center_scaled + margin).astype(int)
    end_pt[0] = min(end_pt[0], width)
    end_pt[1] = min(end_pt[1], height)
    image_scaled = image_padded[start_pt[1]:end_pt[1],
                                start_pt[0]:end_pt[0], :]
    center_scaled -= start_pt
    height, width = image_scaled.shape[:2]
    im_shape = [height, width]

    return {
        # return original too with info.
        'image': image_scaled,
        'im_path': im_path,
        'im_shape': im_shape,
        'center': center_scaled,
        'scale': scale,
        'start_pt': start_pt,
    }


def render_preds(output_path, config, preds, images, images_orig, trim_length,
                 img_size=224):
    """
    Renders a 2x2 video:
    | mesh on input video | mesh on og image space |
    | 2d skel on input    | rotated mesh           |

    Also renders just mesh on og image space.

    images are preprocessed to be [-1, 1]
    """
    renderer = VisRenderer(img_size=img_size)
    renderer_crop = VisRenderer(img_size=img_size)

    # If model doesn't fit in GPU, reduce max_img_size or GPU fraction in Tester
    max_img_size = 720

    output_crop = output_path + '_crop'
    # Final videos
    vid_path = output_path + '.mp4'
    vid_path_crop = output_crop + '.mp4'

    if os.path.exists(vid_path):
        print('Video already exists!')
        return

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_crop):
        os.mkdir(output_crop)

    for j in tqdm(range(trim_length, len(preds['kps']) - trim_length)):
        image = images[j]
        image_og_params = images_orig[j]
        image_og = imread(image_og_params['im_path'])
        cam = preds['cams'][j]
        kps = preds['kps'][j]
        vert = preds['verts'][j]

        skel_og, render_og, rot_og = visualize_img_orig(
            cam=cam,
            kp_pred=kps,
            vert=vert,
            renderer=renderer,
            start_pt=image_og_params['start_pt'],
            scale=image_og_params['scale'],
            proc_img_shape=image_og_params['im_shape'],
            img=((image_og / 255.) - 0.5) * 2,
            mesh_color=config.mesh_color,
            max_img_size=max_img_size,
            no_text=True,
            rotated_view=True,
        )
        plt.imsave(
            fname=os.path.join(output_path,
                               'frame{:06d}.png'.format(j - trim_length)),
            arr=render_og,
        )

        # Render in cropped frame
        skel_crop, rend_crop = visualize_img(
            img=image,
            cam=cam,
            kp_pred=kps,
            vert=vert,
            renderer=renderer_crop,
            mesh_color=config.mesh_color,
        )
        h, w, _ = render_og.shape
        w = w * img_size // h
        render_og = cv2.resize(render_og, (w, img_size))

        padding = np.ones((img_size, np.abs(w - img_size), 3))
        rot_og = cv2.resize(rot_og, (img_size, img_size))
        if w > img_size:
            rot_og = np.hstack((rot_og, padding))
        else:
            render_og = np.hstack((render_og, padding))
        rendered_crop = np.hstack((
            np.vstack((rend_crop, skel_crop)),
            np.vstack((render_og, rot_og))
        ))

        plt.imsave(
            fname=os.path.join(output_crop,
                               'frame{:06d}.png'.format(j - trim_length)),
            arr=rendered_crop,
        )

    print('Converting them to video..')

    make_video(vid_path, output_path)
    make_video(vid_path_crop, output_crop)


def make_video(output_path, img_dir, fps=25):
        """
        output_path is the final mp4 name
        img_dir is where the images to make into video are saved.
        """
        cmd = [
            'ffmpeg',
            '-y',
            '-threads', '16',
            '-framerate', str(fps),
            '-i', '{img_dir}/frame%06d.png'.format(img_dir=img_dir),
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-an',
            # Note that if called as a string, ffmpeg needs quotes around the
            # scale invocation.
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            output_path,
        ]
        print(' '.join(cmd))
        try:
            err = subprocess.call(cmd)
            if err:
                ipdb.set_trace()
        except OSError:
            ipdb.set_trace()
            print('OSError')

