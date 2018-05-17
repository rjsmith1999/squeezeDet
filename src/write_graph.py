from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from config import *
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
	'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
	"""Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
	'out_dir', './data/out/', """Directory to write output graph.""")
tf.app.flags.DEFINE_string(
	'demo_net', 'squeezeDet', """Neural net architecture.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id""")

def main(argv=None):
  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
	  'Selected neural net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlus(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)
      tf.train.write_graph(sess.graph_def, FLAGS.out_dir, FLAGS.demo_net + '.pbtxt')

if __name__ == '__main__':
  tf.app.run()
