from functools import partial

import slim
import tensorflow as tf

from slim import ops
from slim import scopes
from slim.ops import fc as RNN
batch_norm_params = {
    'decay': 0.997,
    'epsilon': 1e-5,
    'scale': True,

}

#ops.NotDifferentiable("ExtractPatches")
# extract_patches_module = tf.load_op_library('extract_patches_gpu.so')
def SRN_arg_scope_tf(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     training = True
                     ):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
    }

    with scopes.arg_scope(
            [slim.ops.conv2d],
            weight_decay=weight_decay,
            activation=tf.nn.relu,
            batch_norm_params=batch_norm_params,
            is_training = training
    ):
        with scopes.arg_scope([slim.ops.fc],
                              weight_decay=weight_decay,
                              activation=tf.nn.relu,
                              is_training = training

                              ):
            with scopes.arg_scope([ops.batch_norm], **batch_norm_params):
                with scopes.arg_scope([ops.max_pool], padding='VALID') as arg_sc:
                    return arg_sc

def align_reference_shape(reference_shape, reference_shape_bb, im, bb):
    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bb) / norm(reference_shape_bb)
    align_mean_shape = (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio + tf.reduce_mean(bb, 0)
    new_size = tf.to_int32(tf.to_float(tf.shape(im)[:2]) / ratio)
    return tf.image.resize_bilinear(tf.expand_dims(im, 0), new_size)[0, :, :, :], align_mean_shape / ratio, ratio

def normalized_rmse(pred, gt_truth):
    norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, 36, :] - gt_truth[:, 45, :])**2), 1))

    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * 68)

def conv_model(inputs, is_training=True, scope=''):

    # summaries or losses.
    net = {}

    with tf.name_scope(scope, 'Conv_lay', [inputs]):
        with scopes.arg_scope(SRN_arg_scope_tf(training=is_training)):
            with scopes.arg_scope([ops.conv2d],
                                  weight_decay=0.0001,
                                  activation=tf.nn.relu,
                                  batch_norm_params=batch_norm_params,
                                  is_training=is_training,
                                  padding='VALID'):
                net['conv_1'] = ops.conv2d(inputs, 32, [7, 7], scope='conv_1')
                net['pool_1'] = ops.max_pool(net['conv_1'], [2, 2])
                net['conv_2'] = ops.conv2d(net['pool_1'], 64, [5, 5], scope='conv_2')
                net['pool_2'] = ops.max_pool(net['conv_2'], [2, 2])
                net['conv_3'] = ops.conv2d(net['pool_2'], 128, [3, 3], scope='conv_3')
                net['pool_3'] = ops.max_pool(net['conv_3'], [2, 2])
    return net

def Non_local(feature,training,scope='NL'):
    batch_size,num, h, w, c = feature.get_shape().as_list()
    feature = tf.reshape(feature, (batch_size, -1, h, w, c))
    with tf.variable_scope(scope):
      with scopes.arg_scope(SRN_arg_scope_tf(training=training)):
        x1 = tf.layers.conv3d(feature, c, [1, 1, 1], name='3D_cnn1')
        x2 = tf.layers.conv3d(feature, c, [1, 1, 1], name='3D_cnn2')
        x3 = tf.layers.conv3d(feature,  c, [1, 1, 1], name='3D_cnn3')
        x1 = tf.reshape(x1, (batch_size, -1, c))
        x2 = tf.reshape(x2, (batch_size, c, -1))
        x3 = tf.reshape(x3, (batch_size, -1, c))
        x4 = tf.matmul(x1, x2)
        x4 = tf.nn.softmax(x4)
        x5 = tf.matmul(x4, x3)
        x5 = tf.reshape(x5, (batch_size,num, h, w, c))
        z_feature = tf.layers.conv3d(x5, c, [1, 1, 1], name='3D_cnn4')
    return z_feature


def GlomFace(images, inits, num_iterations=4, num_patches=68, patch_shape=(42, 42), num_channels=3,reuse = False,training = True):
    batch_size = images.get_shape().as_list()[0]

    is_training = training
    hidden = tf.zeros((batch_size, 1024))

    dx = tf.zeros((batch_size, num_patches, 2))
    endpoints = {}
    dxs = []
    m_module = tf.load_op_library('./extract_patches.so')
    with tf.variable_scope('models', reuse=reuse):
        for step in range(num_iterations):
            with tf.device('/cpu:0'):
                patches = m_module.extract_patches(images, tf.constant(patch_shape), inits+dx)
            patches = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))

            endpoints['patches'] = patches
            with tf.variable_scope('PHM', reuse=step > 0):
                # level1
                net = conv_model(patches, is_training=training)
                ims = net['pool_3']
                batch_num, h, w, c = ims.get_shape().as_list()
                ims_reshape = tf.reshape(ims,(batch_size, batch_num/batch_size, h, w, c))
                ims_all = tf.reshape(ims, (batch_size, -1))

                region0 = ims_reshape[:, 0:9, :, :, :]
                # level3
                net['NL0'] = Non_local(region0, training=training, scope='region0')

                region1 = ims_reshape[:, 9:17, :, :, :]

                net['NL1'] = Non_local(region1, training=training, scope='region1')

                region2 = ims_reshape[:, 17:27, :, :, :]

                net['NL2'] = Non_local(region2, training=training, scope='region2')

                region3 = ims_reshape[:, 27:36, :, :, :]

                net['NL3'] = Non_local(region3, training=is_training, scope='region3')

                region4 = ims_reshape[:, 36:48, :, :, :]

                net['NL4'] = Non_local(region4, training=is_training, scope='region4')

                region5 = ims_reshape[:, 48:68, :, :, :]

                net['NL5'] = Non_local(region5, training=is_training, scope='region5')

                # level4
                group1 = tf.concat([net['NL0'],net['NL1']],1)

                net['NL_g1'] = Non_local(group1, training=is_training, scope='group1')

                group2 = tf.concat([net['NL2'], net['NL4']], 1)

                net['NL_g2'] = Non_local(group2, training=is_training, scope='group2')

                group3 = tf.concat([net['NL3'], net['NL5']], 1)

                net['NL_g3'] = Non_local(group3, training=is_training, scope='group3')

                #level5
                Global_F = tf.concat([net['NL_g1'], net['NL_g2'], net['NL_g3']], 1)
                NL_feature= Non_local(Global_F, training=is_training, scope='whole')

                NL_feature = ims_reshape+NL_feature

                ims_fllaten = tf.reshape(NL_feature, (batch_size, num_patches, -1))
                ims_d = ims_fllaten[:, 0:17, :]
                ims_d = tf.reshape(ims_d, (batch_size, -1))
                ims_t = ims_fllaten[:, 17:39, :]
                ims_t = tf.reshape(ims_t, (batch_size, -1))
                ims_m = ims_fllaten[:, 39:68, :]
                ims_m = tf.reshape(ims_m, (batch_size, -1))
                whole = tf.concat([ims_t,ims_m,ims_d],1)

            with tf.variable_scope('WHM', reuse=step > 0) as scope:
                #level5 whole
                with scopes.arg_scope(SRN_arg_scope_tf(training=training)):
                    hidden = RNN(tf.concat([ims_all, hidden], 1), 1024, scope='rnn', activation=tf.tanh,batch_norm_params = batch_norm_params,is_training=training)
                #level4 group
                res_group1 = slim.ops.fc(tf.concat([whole, hidden], 1), 1024, scope='dis_group1', batch_norm_params=batch_norm_params, is_training=training,weight_decay=0.0001)

                res_group2 = slim.ops.fc(tf.concat([whole, hidden], 1), 1024, scope='dis_group2', batch_norm_params=batch_norm_params, is_training=training,weight_decay=0.0001)

                res_group3 = slim.ops.fc(tf.concat([whole, hidden], 1), 1024, scope='dis_group3', batch_norm_params=batch_norm_params, is_training=training,weight_decay=0.0001)

                with scopes.arg_scope(SRN_arg_scope_tf(training=training)):
                    #level3 region
                    region_triple1 = slim.ops.fc(tf.concat([res_group1, res_group2,res_group3], 1), 512, scope='region_triple1')
                    brow = slim.ops.fc(tf.concat([region_triple1, hidden], 1), 256, scope='brow')

                    region_triple2 = slim.ops.fc(tf.concat([res_group1, res_group2,res_group3], 1), 512, scope='region_triple2')
                    eye = slim.ops.fc(tf.concat([region_triple2, hidden], 1), 256, scope='eye')

                    region_triple3 = slim.ops.fc(tf.concat([res_group1, res_group2,res_group3], 1), 512, scope='region_triple3')
                    nose = slim.ops.fc(tf.concat([region_triple3, hidden], 1), 256, scope='nose')

                    region_triple4 = slim.ops.fc(tf.concat([res_group1, res_group2,res_group3], 1), 512, scope='region_triple4')
                    mouth = slim.ops.fc(tf.concat([region_triple4, hidden], 1), 256, scope='mouth')

                    region_triple5 = slim.ops.fc(tf.concat([res_group1, res_group2,res_group3], 1), 512, scope='region_triple5')
                    cheek_left = slim.ops.fc(tf.concat([region_triple5, hidden], 1), 256, scope='cheek_left')

                    region_triple6 = slim.ops.fc(tf.concat([res_group1, res_group2,res_group3], 1), 512, scope='region_triple6')
                    cheek_right = slim.ops.fc(tf.concat([region_triple6, hidden], 1), 256, scope='cheek_right')

                    #level2 region
                    region_brow = slim.ops.fc(tf.concat([brow, eye], 1), 256, scope='region_brow')
                    region_brow = slim.ops.fc(tf.concat([region_brow, res_group1], 1), 128, scope='region_brow2')

                    region_eye = slim.ops.fc(tf.concat([brow, eye], 1), 256, scope='region_eye')
                    region_eye = slim.ops.fc(tf.concat([region_eye, res_group1], 1), 128, scope='region_eye2')

                    region_nose = slim.ops.fc(tf.concat([nose, mouth], 1), 256, scope='region_nose')
                    region_nose = slim.ops.fc(tf.concat([region_nose, res_group2], 1), 128, scope='region_nose2')

                    region_mouth = slim.ops.fc(tf.concat([nose, mouth], 1), 256, scope='region_mouth')
                    region_mouth = slim.ops.fc(tf.concat([region_mouth, res_group2], 1), 128, scope='region_mouth2')

                    region_cheek_left = slim.ops.fc(tf.concat([cheek_left, cheek_right], 1), 256, scope='region_cheek_left')
                    region_cheek_left = slim.ops.fc(tf.concat([region_cheek_left, res_group3], 1), 128, scope='region_cheek_left2')

                    region_cheek_right = slim.ops.fc(tf.concat([cheek_left, cheek_right], 1), 256, scope='region_cheek_right')
                    region_cheek_right = slim.ops.fc(tf.concat([region_cheek_right, res_group3], 1), 128, scope='region_cheek_right2')
                #levle1 local shape
                with tf.variable_scope('pre', reuse=step > 0):
                    #12 local shapes in fixed order
                    brows_landmarks = slim.ops.fc(tf.concat([region_brow, region_eye], 1), 64, scope='pre_b',batch_norm_params=batch_norm_params, is_training=training,weight_decay=0.0001)
                    brows_landmarks = slim.ops.fc(tf.concat([brows_landmarks, brow], 1), 20, scope='pre_b2',activation=None)
                    brows_landmarks = tf.reshape(brows_landmarks, (batch_size, 10, 2))

                    eyes_landmarks = slim.ops.fc(tf.concat([region_eye, region_brow], 1), 64, scope='pre_e',batch_norm_params=batch_norm_params, is_training=training,weight_decay=0.0001)
                    eyes_landmarks = slim.ops.fc(tf.concat([eyes_landmarks, eye], 1), 24, scope='pre_e2',activation=None)
                    eyes_landmarks = tf.reshape(eyes_landmarks, (batch_size, 12, 2))

                    nose_landmarks = slim.ops.fc(tf.concat([region_nose, region_mouth], 1), 64, scope='pre_n',batch_norm_params=batch_norm_params, is_training=training,weight_decay=0.0001)
                    nose_landmarks = slim.ops.fc(tf.concat([nose_landmarks, nose], 1), 18, scope='pre_n2',activation=None)
                    nose_landmarks = tf.reshape(nose_landmarks, (batch_size, 9, 2))

                    mouth_landmarks = slim.ops.fc(tf.concat([region_mouth, region_nose], 1), 64, scope='pre_mo',batch_norm_params=batch_norm_params, is_training=training,weight_decay=0.0001)
                    mouth_landmarks = slim.ops.fc(tf.concat([mouth_landmarks, mouth], 1), 40, scope='pre_mo2',activation=None)
                    mouth_landmarks = tf.reshape(mouth_landmarks, (batch_size, 20, 2))

                    cheek_left_landmarks = slim.ops.fc(tf.concat([region_cheek_left, region_cheek_right], 1), 64, scope='pre_ll',batch_norm_params=batch_norm_params, is_training=training,weight_decay=0.0001)
                    cheek_left_landmarks = slim.ops.fc(tf.concat([cheek_left_landmarks, cheek_left], 1), 9*2, scope='pre_ll2', activation=None)
                    cheek_left_landmarks = tf.reshape(cheek_left_landmarks, (batch_size, 9, 2))

                    cheek_right_landmarks = slim.ops.fc(tf.concat([region_cheek_right, region_cheek_left], 1), 64, scope='pre_lr',batch_norm_params=batch_norm_params, is_training=training,weight_decay=0.0001)
                    cheek_right_landmarks = slim.ops.fc(tf.concat([cheek_right_landmarks, cheek_right], 1), 8*2, scope='pre_lr2', activation=None)
                    cheek_right_landmarks = tf.reshape(cheek_right_landmarks, (batch_size, 8, 2))

                    offset_global = tf.concat([cheek_left_landmarks, cheek_right_landmarks, brows_landmarks, nose_landmarks, eyes_landmarks, mouth_landmarks], 1)

                endpoints['prediction'] = offset_global
            #concat
            prediction = tf.reshape(offset_global, (batch_size, num_patches, 2))
            dx += prediction
            dxs.append(dx)

    return inits + dx, dxs, endpoints
