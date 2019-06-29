from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import (
    variance_scaling_initializer,
)
import tensorflow.contrib.slim as slim


def get_image_encoder(model_type='resnet'):
    """
    Retrieves encoder fn for image and 3D
    """
    models = {
        'resnet': encoder_resnet
    }
    if model_type in models.keys():
        return models[model_type]
    else:
        print('Unknown image encoder:', model_type)
        exit(1)


def get_hallucinator_model(model_type='fc2_res'):
    models = {
        'fc2_res': fc2_res,
    }
    if model_type in models.keys():
        return models[model_type]
    else:
        print('Unknown predict hal model:', model_type)
        exit(1)


def get_temporal_encoder(model_type='AZ_FC2GN'):
    models = {
        'AZ_FC2GN': az_fc2_groupnorm,
    }
    if model_type in models.keys():
        return models[model_type]
    else:
        print('Unknown temporal encoder:', model_type)
        exit(1)


# Functions for image encoder.

def encoder_resnet(x, is_training=True, weight_decay=0.001, reuse=False):
    """
    Resnet v2-50
    Assumes input is [batch, height_in, width_in, channels]!!
    Input:
    - x: N x H x W x 3
    - weight_decay: float
    - reuse: bool->True if test

    Outputs:
    - cam: N x 3
    - Pose vector: N x 72
    - Shape vector: N x 10
    - variables: tf variables
    """
    from tensorflow.contrib.slim.python.slim.nets import resnet_v2
    with tf.name_scope('Encoder_resnet', [x]):
        with slim.arg_scope(
                resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            net, end_points = resnet_v2.resnet_v2_50(
                x,
                num_classes=None,
                is_training=is_training,
                reuse=reuse,
                scope='resnet_v2_50')
            net = tf.squeeze(net, axis=[1, 2])
    variables_scope = 'resnet_v2_50'
    return net, variables_scope


def encoder_fc3_dropout(x,
                        num_output=85,
                        is_training=True,
                        reuse=False,
                        name='3D_module'):
    """
    3D inference module. 3 MLP layers (last is the output)
    With dropout  on first 2.
    Input:
    - x: N x [|img_feat|, |3D_param|]
    - reuse: bool

    Outputs:
    - 3D params: N x num_output
      if orthogonal:
           either 85: (3 + 24*3 + 10) or 109 (3 + 24*4 + 10) for factored
           axis-angle representation
      if perspective:
          86: (f, tx, ty, tz) + 24*3 + 10, or 110 for factored axis-angle.
    - variables: tf variables
    """
    with tf.variable_scope(name, reuse=reuse) as scope:
        net = slim.fully_connected(x, 1024, scope='fc1')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
        net = slim.fully_connected(net, 1024, scope='fc2')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        small_xavier = variance_scaling_initializer(
            factor=.01, mode='FAN_AVG', uniform=True)
        net = slim.fully_connected(
            net,
            num_output,
            activation_fn=None,
            weights_initializer=small_xavier,
            scope='fc3')

    variables = tf.contrib.framework.get_variables(scope)
    return net, variables


# Functions for f_{movie strip}.

def az_fc2_groupnorm(is_training, net, num_conv_layers):
    """
    Each block has 2 convs.
    So:
    norm --> relu --> conv --> norm --> relu --> conv --> add.
    Uses full convolution.


    Args:

    """
    for i in range(num_conv_layers):
        net = az_fc_block2(
            net_input=net,
            num_filter=2048,
            kernel_width=3,
            is_training=is_training,
            use_groupnorm=True,
            name='block_{}'.format(i),
        )
    return net


def az_fc_block2(net_input, num_filter, kernel_width, is_training,
                 use_groupnorm=False, name=None):
    """
    Full convolutions not separable!!
    same as az_fc_block, but residual connections is proper Kaiming style
    of BN -> Relu -> Weight -> BN -> Relu -> Weight -> Add
    """
    # NTC -> NT1C
    net_input_expand = tf.expand_dims(net_input, axis=2)
    if use_groupnorm:
        # group-norm
        net_norm = tf.contrib.layers.group_norm(
            net_input_expand,
            channels_axis=-1,
            reduction_axes=(-3, -2),
            scope='AZ_FC_block_preact_gn1' + name,
            reuse=None,
        )
    else:
        # batchnorm
        net_norm = tf.contrib.layers.batch_norm(
            net_input_expand,
            scope='AZ_FC_block_preact_bn1' + name,
            reuse=None,
            is_training=is_training,
        )
    # relu
    net_relu = tf.nn.relu(net_norm)
    # weight
    net_conv1 = tf.contrib.layers.conv2d(
        inputs=net_relu,
        num_outputs=num_filter,
        kernel_size=[kernel_width, 1],
        stride=1,
        padding='SAME',
        data_format='NHWC',  # was previously 'NCHW',
        rate=1,
        activation_fn=None,
        scope='AZ_FC_block2_conv1' + name,
        reuse=None,
    )
    # norm
    if use_groupnorm:
        # group-norm
        net_norm2 = tf.contrib.layers.group_norm(
            net_conv1,
            channels_axis=-1,
            reduction_axes=(-3, -2),
            scope='AZ_FC_block_preact_gn2' + name,
            reuse=None,
        )
    else:
        net_norm2 = tf.contrib.layers.batch_norm(
            net_conv1,
            scope='AZ_FC_block_preact_bn2' + name,
            reuse=None,
            is_training=is_training,
        )

    # relu
    net_relu2 = tf.nn.relu(net_norm2)
    # weight
    small_xavier = variance_scaling_initializer(
        factor=.001, mode='FAN_AVG', uniform=True)

    net_final = tf.contrib.layers.conv2d(
        inputs=net_relu2,
        num_outputs=num_filter,
        kernel_size=[kernel_width, 1],
        stride=1,
        padding='SAME',
        data_format='NHWC',
        rate=1,
        activation_fn=None,
        weights_initializer=small_xavier,
        scope='AZ_FC_block2_conv2' + name,
        reuse=None,
    )

    # NT1C -> NTC
    net_final = tf.squeeze(net_final, axis=2)
    # skip connection
    residual = tf.add(net_final, net_input)

    return residual


# Functions for f_3D.

def batch_pred_omega(input_features, batch_size, is_training, num_output,
                     omega_mean, sequence_length, scope, predict_delta_keys=(),
                     use_delta_from_pred=False, use_optcam=False):
    """
    Given B x T x * inputs, computes IEF on them by batching them
    as BT x *.

    if use_optcam is True, only outputs 72 or 82 dims.
    and appends fixed camera [1,0,0]
    """
    # run in batch
    # omega_mean comes in as shape: BT x 85
    input_features_reshape = tf.reshape(input_features,
                                        (batch_size * sequence_length, -1))
    omega_pred, delta_predictions = call_hmr_ief(
        phi=input_features_reshape,
        omega_start=omega_mean,
        scope=scope,
        num_output=num_output,
        is_training=is_training,
        predict_delta_keys=predict_delta_keys,
        use_delta_from_pred=use_delta_from_pred,
        use_optcam=use_optcam,
    )
    omega_pred = tf.reshape(
        omega_pred,
        (batch_size, sequence_length, num_output)
    )
    new_delta_predictions = {}
    for delta_t, prediction in delta_predictions.items():
        new_delta_predictions[delta_t] = tf.reshape(
            prediction,
            (batch_size, sequence_length, num_output)
        )
    return omega_pred, new_delta_predictions


def fc2_res(phi, name='fc2_res'):
    """
    Converts pretrained (fixed) resnet features phi into movie strip.

    This applies 2 fc then add it to the orig as residuals.

    Args:
        phi (B x T x 2048): Image feature.
        name (str): Scope.

    Returns:
        Phi (B x T x 2048): Hallucinated movie strip.
    """
    with tf.variable_scope(name, reuse=False):
        net = slim.fully_connected(phi, 2048, scope='fc1')
        net = slim.fully_connected(net, 2048, scope='fc2')
        small_xavier = variance_scaling_initializer(
            factor=.001, mode='FAN_AVG', uniform=True)
        net_final = slim.fully_connected(
            net,
            2048,
            activation_fn=None,
            weights_initializer=small_xavier,
            scope='fc3'
        )
        new_phi = net_final + phi
    return new_phi


def call_hmr_ief(phi, omega_start, scope, num_output=85, num_stage=3,
                 is_training=True, predict_delta_keys=(),
                 use_delta_from_pred=False, use_optcam=True):
    """
    Wrapper for doing HMR-style IEF.

    If predict_delta, then also makes num_delta_t predictions forward and
    backward in time, with each step of delta_t.

    Args:
        phi (Bx2048): Image features.
        omega_start (Bx85): Starting Omega as input to first IEF.
        scope (str): Name of scope for reuse.
        num_output (int): Size of output.
        num_stage (int): Number of iterations for IEF.
        is_training (bool): If False, don't apply dropout.
        predict_delta_keys (iterable): List of keys for delta_t.
        use_delta_from_pred (bool): If True, initializes delta prediction from
            current frame prediction.
        use_optcam (bool): If True, uses [1, 0, 0] for cam.

    Returns:
        Final theta (Bx{num_output})
        Deltas predictions (List of outputs)
    """
    theta_here = hmr_ief(
        phi=phi,
        omega_start=omega_start,
        scope=scope,
        num_output=num_output,
        num_stage=num_stage,
        is_training=is_training
    )

    # Delta only needs to do cam/pose, no shape!
    if use_optcam:
        num_output_delta = 72
    else:
        num_output_delta = 3 + 72

    deltas_predictions = {}
    for delta_t in predict_delta_keys:
        if delta_t == 0:
            # This should just be the normal IEF.
            continue
        elif delta_t > 0:
            scope_delta = scope + '_future{}'.format(delta_t)
        elif delta_t < 0:
            scope_delta = scope + '_past{}'.format(abs(delta_t))

        omega_start_delta = theta_here if use_delta_from_pred else omega_start
        # append this later.
        beta = omega_start_delta[:, -10:]

        if use_optcam:
            # trim the first 3D camera + last shpae
            omega_start_delta = omega_start_delta[:, 3:3 + num_output_delta]
        else:
            omega_start_delta = omega_start_delta[:, :num_output_delta]

        delta_pred = hmr_ief(
            phi=phi,
            omega_start=omega_start_delta,
            scope=scope_delta,
            num_output=num_output_delta,
            num_stage=num_stage,
            is_training=is_training
        )
        if use_optcam:
            # Add camera + shape
            scale = tf.ones([delta_pred.shape[0], 1])
            trans = tf.zeros([delta_pred.shape[0], 2])
            delta_pred = tf.concat([scale, trans, delta_pred, beta], 1)
        else:
            delta_pred = tf.concat([delta_pred[:, :75], beta], 1)

        deltas_predictions[delta_t] = delta_pred

    return theta_here, deltas_predictions


def hmr_ief(phi, omega_start, scope, num_output=85, num_stage=3,
            is_training=True):
    """
    Runs HMR-style IEF.

    Args:
        phi (Bx2048): Image features.
        omega_start (Bx85): Starting Omega as input to first IEF.
        scope (str): Name of scope for reuse.
        num_output (int): Size of output.
        num_stage (int): Number of iterations for IEF.
        is_training (bool): If False, don't apply dropout.

    Returns:
        Final theta (Bx{num_output})
    """
    with tf.variable_scope(scope):
        theta_prev = omega_start
        theta_here = None

        for _ in range(num_stage):
            # ---- Compute outputs
            state = tf.concat([phi, theta_prev], 1)
            delta_theta, _ = encoder_fc3_dropout(
                state,
                is_training=is_training,
                num_output=num_output,
                reuse=tf.AUTO_REUSE
            )
            # Compute new theta
            theta_here = theta_prev + delta_theta

            # Finally update to end iteration.
            theta_prev = theta_here

    return theta_here
