from datetime import datetime
import Glom_data_provider as data_provider
from menpo.shape.pointcloud import PointCloud
import Glom_model as Glom
import numpy as np
import os.path
import slim
import tensorflow as tf
import time
import utils
import menpo
import scipy.io as sio
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as tf_variables
import os        #feilong-----------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#-------------------------------------------------

#ignore  differentiable for extract_patches op
ops.NotDifferentiable("ExtractPatches")
ops.NotDifferentiable("ResizeBilinearGrad")
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('batch_size', 100, """The batch size to use.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """How many preprocess threads to use.""")
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 200000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('train_device', '/gpu:0', """Device to train with.""")
tf.app.flags.DEFINE_string('datasets', ':'.join(
    (
        'databases/lfpw/trainset/*.png',
        'databases/afw/*.jpg',
        'databases/helen/trainset/*.jpg',
    )),
                           """Directory where to write event logs """
                           """and checkpoint.""")
# tf.app.flags.DEFINE_string('real_data', ':'.join(
#     (
#         'databases/COFW_color/trainset/*.jpg',
#
#     )),
#                            """Directory where to write event logs """
#                            """and checkpoint.""")
tf.app.flags.DEFINE_float('image_size', 198

                          , 'The extracted patch size')
tf.app.flags.DEFINE_float('crop_size', 224.

                          , 'The extracted patch size')
tf.app.flags.DEFINE_integer('patch_size', 40

                            , 'The extracted patch size')
# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999
gpu_num = 4
def train(scope=''):

    # H-LIU: reallocating gpu memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    """Train on dataset for a number of steps."""
    with tf.Graph().as_default():

        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        train_dirs = FLAGS.datasets.split(':')
        # real_data = FLAGS.real_data.split(':')

        # Calculate the learning rate schedule.
        decay_steps = 3000
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(lr)


        # Override the number of preprocessing threads to account for the increased
        # number of GPU towers.
        num_preprocess_threads = FLAGS.num_preprocess_threads

        _images, _shapes, _reference_shape, pca_model, center_space, labels = \
            data_provider.load_images(train_dirs)
        # rela_images  = \
        #     data_provider.load_data(real_data,_reference_shape)
        reference_shape = tf.constant(_reference_shape,
                                      dtype=tf.float32,
                                      name='initial_shape')
        image_shape = _images[0].shape
        # real_shape = rela_images[0].shape
        lms_shape = _shapes[0].points.shape

        # patch = np.random.rand((3,50,50))
        def get_random_sample(rotation_stddev=10):
            image_patch = menpo.image.Image(np.random.rand(3, 256, 256))
            idx = np.random.randint(low=0, high=len(_images))
            candidate_idx = np.random.randint(low=0, high=50)
            while labels[idx] != candidate_idx:
                idx = np.random.randint(low=0, high=len(_images))

            im = menpo.image.Image(_images[idx].transpose(2, 0, 1), copy=False)
            lms = _shapes[idx]

            im.landmarks['PTS'] = lms

            meanshape = PointCloud(_reference_shape)
            im.landmarks['random'] = meanshape

            if np.random.rand() < .5:
                im = utils.mirror_image(im)

            if np.random.rand() < .5:
                theta = np.random.normal(scale=rotation_stddev)
                rot = menpo.transform.rotate_ccw_about_centre(lms, theta)
                im = im.warp_to_shape(im.shape, rot)
            if np.random.rand() < .5:
                occlusion_width = np.random.randint(30, 130)
                occlusion_hight = np.random.randint(30, 130)

                center_w = np.random.randint(int(0.5*occlusion_width)+1, FLAGS.crop_size-int(0.5*occlusion_width)-1)
                center_h = np.random.randint(int(0.5*occlusion_hight)+1, FLAGS.crop_size-int(0.5*occlusion_hight)-1)

                center = PointCloud(np.array([[center_w, center_h]]).reshape(1,2))

                images_patchs = image_patch.extract_patches(center,(occlusion_width,occlusion_hight))
                # images_patchs = menpo.image.Image(np.random.rand(3, occlusion_width, occlusion_hight))
                im = utils.set_patches(im,images_patchs,center)
                # print im.shape
            # if np.random.rand() < .5:
            #     init_index = np.random.randint(low=0, high=len(_shapes))
            #     init = _shapes[init_index]
            #     im.landmarks['random'] = init

            #pixels = im.pixels.transpose(1, 2, 0).astype('float32')
            pixels_occ = im.pixels.transpose(1, 2, 0).astype('float32')
            # real_pixels = real_im.pixels.transpose(1, 2, 0).astype('float32')
            shape = im.landmarks['PTS'].lms.points.astype('float32')
            init = im.landmarks['random'].lms.points.astype('float32')
            # print pixels_occ
            # print pixels_occ.shape
            return pixels_occ, shape, init

        image, shape , random_init= tf.py_func(get_random_sample, [],
                                               [tf.float32, tf.float32, tf.float32], stateful=True)

        initial_shape = data_provider.random_shape(shape, reference_shape,
                                                   pca_model)
        image.set_shape(image_shape)
        # ims_oc.set_shape(ims_oc)
        shape.set_shape(lms_shape)
        initial_shape.set_shape(lms_shape)
        # initial_shape = initial_shape*(198./224.)

        do_scale = tf.random_uniform([1])*0.4+0.885
        bbx_jitter = tf.random_uniform([1])
        image_height = tf.to_int32(tf.to_float(FLAGS.crop_size) * do_scale[0])
        image_width = tf.to_int32(tf.to_float(FLAGS.crop_size) * do_scale[0])
        image = tf.image.resize_images(image, tf.stack([image_height, image_width]))
        shape = shape * do_scale
        initial_shape = initial_shape * do_scale

        target_h = tf.to_int32(FLAGS.image_size)
        target_w = tf.to_int32(FLAGS.image_size)
        offset_h = tf.to_int32((tf.to_int32(image_height) - target_h) / 2)
        offset_w = tf.to_int32((tf.to_int32(image_width) - target_w) / 2)
        offset_h = tf.to_int32(tf.to_float(offset_h) * bbx_jitter[0])
        offset_w = tf.to_int32(tf.to_float(offset_w) * bbx_jitter[0])
        image = tf.image.crop_to_bounding_box(
            image, offset_h, offset_w, target_h, target_w)
        shape = shape - tf.to_float(tf.stack([offset_h, offset_w]))
        initial_shape = initial_shape - tf.to_float(tf.stack([offset_h, offset_w]))

        image = data_provider.distort_color(image)

        images, lms, inits = tf.train.shuffle_batch([image, shape, initial_shape],
                                            FLAGS.batch_size,
                                            capacity=5000,
                                            min_after_dequeue=1000,
                                            enqueue_many=False,
                                            num_threads=num_preprocess_threads,
                                            name='batch')
        print('Defining model...')
        # image_shape = tf.shape(ims_os)
        # resize_images = tf.image.resize_images(ims_os,(256,256))
        # real_ = tf.image.resize_images(real_ims,(256,256))
        # g_images = models.generator(ims_os)
        # _, d_real = models.discriminator(real_ims)
        # _, d_fake = models.discriminator(g_images,reuse=True)
        # # Retain the summaries from the final tower.
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        # # g_images = tf.image.resize_images(g_images,(128,128))
        # loss_s = tf.reduce_mean(tf.image.ssim(images, g_images,0.05))*150
        # # real_ims = tf.image.resize_images(real_ims,(image_shape[1],image_shape[2]))
        tower_grade = []
        tower_loss = []
        images_split = tf.split(images,gpu_num)
        inits_split = tf.split(inits,gpu_num)
        lms_split = tf.split(lms, gpu_num)
        with tf.variable_scope(tf.get_variable_scope()):
            for d in range(gpu_num):
                with tf.device('/gpu:%s' % d):
                        with tf.name_scope('%s_%s'%('tower',d)):
                            predictions, dxs, _ = Glom.model(images_split[d], inits_split[d], patch_shape=(FLAGS.patch_size, FLAGS.patch_size))
                            detector_loss = 0
                            tf.get_variable_scope().reuse_variables()
                            for i, dx in enumerate(dxs):
                                norm_error = Glom.normalized_rmse(dx + inits_split[d], lms_split[d])
                                tf.summary.histogram('errors', norm_error)
                                loss = tf.reduce_mean(norm_error)
                                detector_loss += loss
                            # tf.losses.add_loss(detector_loss)
                            # total_loss = tf.losses.get_total_loss()
                        with tf.variable_scope('loss'):
                            grads = opt.compute_gradients(detector_loss)
                            tower_grade.append(grads)
                            tower_loss.append(detector_loss)

            mean_loss = tf.stack(axis=0,values=tower_loss)
            mean_loss = tf.reduce_mean(mean_loss,0)
            mean_grads =utils.average_gradients(tower_grade)
                # for i, dx in enumerate(dxs_g):
                #     norm_error = mdm_model.normalized_rmse(dx + inits, lms)
                #     # tf.summary.histogram('errors', norm_error)
                #     loss = tf.reduce_mean(norm_error)
                #     detector_loss += loss
                #     summaries.append(tf.summary.scalar('losses/step_{}'.format(i),
                #                                        loss))

                # Calculate the gradients for the batch of data
                # capped_gvs = []

            # for grad, var in grads:
            #     if grad != None:
            #         capped_gvs.append((tf.clip_by_value(grad, -0.05, 0.05), var))

        # summaries.append(tf.summary.scalar('losses/total_loss', total_loss))
        #
        #
        # summary = tf.summary.image('images',
        #                            tf.concat([images],2),
        #                            5)
        # summaries.append(tf.summary.histogram('dx', predictions - inits))
        #
        # summaries.append(summary)

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                              scope)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in mean_grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name +
                                                      '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        with tf.control_dependencies(batchnorm_updates):
            apply_gradient_op = opt.apply_gradients(mean_grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

            # Track the moving averages of all trainable variables.
            # Note that we maintain a "double-average" of the BatchNormalization
            # global statistics. This is more complicated then need be but we employ
            # this for backward-compatibility with our previous models.
            variable_averages = tf.train.ExponentialMovingAverage(
                MOVING_AVERAGE_DECAY, global_step)

            # Another possibility is to use tf.slim.get_variables().
            variables_to_average = (
                    tf.trainable_variables() + tf.moving_average_variables())
            variables_averages_op = variable_averages.apply(variables_to_average)

            # Group all updates to into a single train op.
            # NOTE: Currently we are not using batchnorm in MDM.
            batchnorm_updates_op = tf.group(*batchnorm_updates)

            train_op = tf.group(apply_gradient_op, variables_averages_op,
                                batchnorm_updates_op)


        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())


        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)
        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config = config)
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        print('Initializing variables...')
        sess.run(init)
        print('Initialized variables.')


        if FLAGS.pretrained_model_checkpoint_path:
            # assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir)

        print('Starting training...')
        for step in range(FLAGS.max_steps):

            start_time = time.time()
            _, loss_value = sess.run([train_op, mean_loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, duration))

            save_fn = '50loss.mat'
            b = np.array([[step, loss_value]])
            if step == 0:
                a = b
            else:
                a = np.concatenate((a, b), axis=0)
            sio.savemat(save_fn, {'loss': a})


            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()
