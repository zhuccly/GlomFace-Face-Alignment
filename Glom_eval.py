"""A library to evaluate MDM on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from pathlib import Path
import Glom_data_provider as data_provider
import math
import menpo
import matplotlib
import Glom_model as net
import numpy as np
import os.path
import tensorflow as tf
import time
import utils
import menpo.io as mio

# Do not use a gui toolkit for matlotlib.
matplotlib.use('Agg')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'ckpt/eval/',
                           """Directory where to write event logs.""")
# tf.app.flags.DEFINE_string('checkpoint_dir', '/home/hliu/data2/CongcongZhu/mdm/ckpt/pre_train',
#                            """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_string('checkpoint_dir', '/media/Store4/zcc/Glom68/Glom/ckpt/GF_model',
                           """Directory where to read model checkpoints.""")
# tf.app.flags.DEFINE_string('checkpoint_dir', 'ckpt/pre_train',
#                            """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 10,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 507,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_string('dataset_path', 'databases/COFW_color/testset/*.jpg',
                           """The dataset path to evaluate.""")
tf.app.flags.DEFINE_string('device', '/gpu:0', 'the device to eval on.')
tf.app.flags.DEFINE_integer('batch_size', 1, """The batch size to use.""")
tf.app.flags.DEFINE_integer('patch_size', 42
                            , 'The extracted patch size')
MOVING_AVERAGE_DECAY = 0.9999
num_iteration = 4
def plot_ced(errors, method_names=['MDM']):
    from matplotlib import pyplot as plt
    from menpofit.visualize import plot_cumulative_error_distribution
    import numpy as np
    # plot the ced and store it at the root.
    fig = plt.figure()
    fig.add_subplot(111)
    plot_cumulative_error_distribution(errors, legend_entries=method_names,
                                       error_range=(0, 0.09, 0.005))
    # shift the main graph to make room for the legend
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    return data

def _eval_once(saver, summary_writer, rmse_op, summary_op,g):
  """Runs Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    rmse_op: rmse_op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
        # flops = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.float_operation())
        # params = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        # print('FLOPs: {}; Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

      else:
        # Restores from checkpoint with relative path.
        # saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
        #                                  ckpt.model_checkpoint_path))
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                         ckpt.model_checkpoint_path))
        # flops = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.float_operation())
        # params = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        # print('FLOPs: {}; Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      # Counts the number of correct predictions.
      errors = []
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.dataset_path))
      start_time = time.time()
      print('%s:'% datetime.now())
      while step < num_iter and not coord.should_stop():
        rmse = sess.run(rmse_op)
        errors.append(rmse)
        step += 1
        if step % 20 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()
      errors = np.vstack(errors).ravel()
      print(errors)
      # save_fn = 'Glom_cofw68.mat'
      # sio.savemat(save_fn, {'errors': errors})

      mean_rmse = errors.mean()
      fr_at_1 = (errors < .08).mean()
      auc_at_05 = (errors < .05).mean()
      ced_image = plot_ced([errors.tolist()])
      ced_plot = sess.run(tf.summary.merge([tf.summary.image('ced_plot', ced_image[None, ...])]))
      print('Errors', errors.shape)
      print('%s: mean_rmse = %.4f, fr @ 0.05 = %.4f, fr @ 0.08 = %.4f [%d examples]' %
            (datetime.now(), errors.mean(), auc_at_05, fr_at_1, total_sample_count))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      # summary.value.add(tag='AUC @ 0.08', simple_value=float(fr_at_1))
      # summary.value.add(tag='AUC @ 0.05', simple_value=float(auc_at_05))
      summary.value.add(tag='Mean RMSE', simple_value=float(mean_rmse))
      summary_writer.add_summary(ced_plot, global_step)
      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
def flip_predictions(predictions, shapes):
    flipped_preds = []
    
    for pred, shape in zip(predictions, shapes):
        pred = menpo.shape.PointCloud(pred)
        pred = utils.mirror_landmarks_68(pred, shape)
        flipped_preds.append(pred.points)
    return np.array(flipped_preds, np.float32)

def evaluate(dataset_path):
    """Evaluate model on Dataset for a number of steps."""
    train_dir = Path(FLAGS.checkpoint_dir)
    # reference_shape = mio.import_pickle(train_dir / 'reference_shape.pkl')
    reference_shape = mio.import_pickle(train_dir / 'reference_shape.pkl')
    print([dataset_path])
    with tf.Graph().as_default() as g, tf.device('/cpu:0'):
        images, gt_truth, inits, _ , bb,bbx = data_provider.batch_inputs(
                [dataset_path], reference_shape,
                batch_size=FLAGS.batch_size, is_training=False)
        mirrored_images, _, mirrored_inits, shapes, _,_ = data_provider.batch_inputs(
            [dataset_path], reference_shape,
            batch_size=FLAGS.batch_size, is_training=False, mirror_image=True)
        print('Loading model...')
    # Build a Graph that computes the logits predictions from the
    # inference model.
        with tf.device(FLAGS.device):
            patch_shape = (FLAGS.patch_size, FLAGS.patch_size)
            pred, _,_ = net.GlomFace(images, inits,num_iterations= num_iteration,patch_shape=patch_shape,training=False)

            tf.get_variable_scope().reuse_variables()

            pred_mirrored, _,_ = net.GlomFace(
            mirrored_images, mirrored_inits, num_iterations=num_iteration,patch_shape=patch_shape,training=False)

        pred_images, = tf.py_func(utils.batch_draw_landmarks_68,
                [images, pred], [tf.float32])
        gt_images, = tf.py_func(utils.batch_draw_landmarks_68,
                [images, gt_truth], [tf.float32])
        # init_images, = tf.py_func(utils.batch_draw_landmarks,
        #         [images, inits], [tf.float32])

        summaries = []
        summaries.append(tf.summary.image('images',
            tf.concat([gt_images, pred_images],2), 135))

        avg_pred = pred + tf.py_func(flip_predictions, (pred_mirrored, shapes), (tf.float32, ))[0]
        # avg_pred = pred

        avg_pred /= 2.
        # NME
        norm_error = net.normalized_rmse(avg_pred, gt_truth)
        # norm_error = net.bbs_rmse(avg_pred, gt_truth,bbx)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge(summaries)

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                                graph_def=graph_def)

        while True:
          _eval_once(saver, summary_writer, norm_error, summary_op,g)
          if FLAGS.run_once:
            break
          time.sleep(FLAGS.eval_interval_secs)

if __name__ == '__main__':
  evaluate(FLAGS.dataset_path)
