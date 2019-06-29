"""
A wrapper for all the prediction functionality.

Caches all the predictions when evaluating.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os
import os.path as osp
import re
from time import time

import numpy as np


PRED_DIR = 'predictions_cache'


def get_pred_path_name(load_path, tf_path, p_id, pred_dir=PRED_DIR,
                       incl_verts=False):
    """
    Gets path name to store cached predictions. If directory doesn't exist,
    build it automatically.

    File structure:
        +-- PRED_DIR
           +-- load_dir
               +-- per dataset per tfrecord preds pickle.
               +-- per dataset per tfrecord eval pickle.
               \-- results.json
    Args:
        load_path (str): Model load path.
        tf_path (str): Path to tfrecord.
        p_id (int): Index per person (when there are multiple tubes).
        pred_dir (str): Directory to store all predictions.
        incl_verts (bool): If True, appends '_verts' suffix.

    Returns:
        Path to prediction pickle.
    """
    vid_id = os.path.basename(tf_path).replace('.tfrecord', '')
    # Dataset is 2 parent dirs up.
    dataset = os.path.basename(os.path.dirname(os.path.dirname(tf_path)))
    load_dir = os.path.basename(load_path)
    verts = '-verts' if incl_verts else ''
    output_name = '{dataset}-{vid}-P{p_id}{verts}.pkl'.format(
        dataset=dataset,
        vid=vid_id,
        p_id=p_id,
        verts=verts,
    )
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    out_dir = os.path.join(pred_dir, load_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    output_path = os.path.join(out_dir, output_name)
    return output_path, output_name


def get_result_path_name(split, load_path, pred_mode, datasets,
                         pred_dir=PRED_DIR):
    """
    Gets path name to store computed evaluations.
    """
    load_dir = os.path.basename(load_path)
    output_name = 'results_{split}_{pred_mode}_{datasets}.json'.format(
        split=split,
        pred_mode=pred_mode,
        datasets=('-'.join(datasets))
    )
    output_path = os.path.join(
        pred_dir,
        load_dir,
        output_name
    )
    return output_path


def get_eval_path_name(load_path, pred_mode, tf_path, p_id, pred_dir=PRED_DIR,
                       min_visible=0):
    """
    Gets path name to store computed evaluations.
    """
    vid_id = os.path.basename(tf_path).replace('.tfrecord', '')
    load_dir = os.path.basename(load_path)
    output_name = 'results_{pred_mode}_{vid_id}_P{p_id}'.format(
        pred_mode=pred_mode,
        vid_id=vid_id,
        p_id=p_id
    )
    if min_visible > 0:
        output_name += '_min-vis{}'.format(min_visible)
    output_path = os.path.join(
        pred_dir,
        load_dir,
        output_name
    )
    return output_path + '.pkl'


def split_preds(preds):
    """
    Remove the verts from the pred dict.
    """
    preds_dict = {}
    verts_dict = {}
    for k, v in preds.items():
        if 'vert' in k:
            verts_dict[k] = v
        else:
            preds_dict[k] = v
    return preds_dict, verts_dict


def get_predictions(model, images, load_path, tf_path, p_id, pred_dir=PRED_DIR,
                    incl_verts=False,):
    """
    If predictions exist, load from pickle. Otherwise, makes the predictions.
    """
    t0 = time()
    pred_path, output_name = get_pred_path_name(
        load_path=load_path,
        tf_path=tf_path,
        p_id=p_id,
        pred_dir=pred_dir,
        incl_verts=False,
    )
    vert_path, _ = get_pred_path_name(
        load_path=load_path,
        tf_path=tf_path,
        p_id=p_id,
        pred_dir=pred_dir,
        incl_verts=True,
    )

    if osp.exists(pred_path) and (not incl_verts or osp.exists(vert_path)):
        print('Loading existing predictions!')
        with open(pred_path, 'rb') as f:
            preds = pickle.load(f)
        if incl_verts:
            with open(vert_path, 'rb') as f:
                preds.update(pickle.load(f))
    else:
        print('Time to compute the predictions D:<')
        if np.max(images) > 1.1:
            # Quick sanity check.
            images = (np.array(images) / 255) * 2 - 1
        preds = model.predict_all_images(images)
        preds.update({
            'tf_path': tf_path,
            'p_id': p_id,
        })
        preds, verts = split_preds(preds)
        with open(pred_path, 'wb') as f:
            pickle.dump(preds, f)
        if incl_verts:
            preds.update(verts)
            with open(vert_path, 'wb') as f:
                pickle.dump(verts, f)
    print('Prediction time:', time() - t0)
    return preds
