import tensorflow as tf
import os
from utils import lrelu, nameop, tbn, obn

def correspondence_loss(b1, b2):
    """
    The correspondence loss.

    :param b1: a tensor representing the object in the graph of the current minibatch from domain one
    :param b2: a tensor representing the object in the graph of the current minibatch from domain two
    :returns a scalar tensor of the correspondence loss
    """
    domain1cols = [0]
    domain2cols = [0]
    loss = tf.constant(0.)
    for c1, c2 in zip(domain1cols, domain2cols):
        loss += tf.reduce_mean((b1[:, c1] - b2[:, c2])**2)

    return loss

class MAGAN(object):
    """The MAGAN model."""

    def __init__(self,
        dim_b1,
        dim_b2,
        activation=lrelu,
        learning_rate=.001,
        restore_folder='',
        limit_gpu_fraction=1.,
        no_gpu=False,
        nfilt=64):
        """Initialize the model."""
        self.dim_b1 = dim_b1
        self.dim_b2 = dim_b2
        self.activation = activation
        self.learning_rate = learning_rate
        self.iteration = 0

        if restore_folder:
            self._restore(restore_folder)
            return

        self.xb1 = tf.placeholder(tf.float32, shape=[None, self.dim_b1], name='xb1')
        self.xb2 = tf.placeholder(tf.float32, shape=[None, self.dim_b2], name='xb2')

        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        self._build()
        self.init_session(limit_gpu_fraction=limit_gpu_fraction, no_gpu=no_gpu)
        self.graph_init(self.sess)

    def init_session(self, limit_gpu_fraction=.4, no_gpu=False):
        """Initialize the session."""
        if no_gpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
        elif limit_gpu_fraction:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
            config = tf.ConfigProto(gpu_options=gpu_options)
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()

    def graph_init(self, sess=None):
        """Initialize graph variables."""
        if not sess: sess = self.sess

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())

        return self.saver

    def save(self, iteration=None, saver=None, sess=None, folder=None):
        """Save the model."""
        if not iteration: iteration = self.iteration
        if not saver: saver = self.saver
        if not sess: sess = self.sess
        if not folder: folder = self.save_folder

        savefile = os.path.join(folder, 'MAGAN')
        saver.save(sess, savefile, write_meta_graph=True)
        print("Model saved to {}".format(savefile))

    def _restore(self, restore_folder):
        """Restore the model from a saved checkpoint."""
        tf.reset_default_graph()
        self.init_session()
        ckpt = tf.train.get_checkpoint_state(restore_folder)
        self.saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print("Model restored from {}".format(restore_folder))

    def _build(self):
        """Construct the DiscoGAN operations."""
        self.G12 = Generator(self.dim_b2, name='G12')
        self.Gb2 = self.G12(self.xb1)
        self.Gb2 = nameop(self.Gb2, 'Gb2')

        self.G21 = Generator(self.dim_b1, name='G21')
        self.Gb1 = self.G21(self.xb2)
        self.Gb1 = nameop(self.Gb1, 'Gb1')

        self.xb2_reconstructed = self.G12(self.Gb1, reuse=True)
        self.xb1_reconstructed = self.G21(self.Gb2, reuse=True)
        self.xb1_reconstructed = nameop(self.xb1_reconstructed, 'xb1_reconstructed')
        self.xb2_reconstructed = nameop(self.xb2_reconstructed, 'xb2_reconstructed')

        self.D1 = Discriminator(name='D1')
        self.D2 = Discriminator(name='D2')

        self.D1_probs_z = self.D1(self.xb1)
        self.D1_probs_G = self.D1(self.Gb1, reuse=True)

        self.D2_probs_z = self.D2(self.xb2)
        self.D2_probs_G = self.D2(self.Gb2, reuse=True)

        self.D1_probs_xrecon = self.D2(self.xb1_reconstructed, reuse=True)
        self.D2_probs_xrecon = self.D2(self.xb2_reconstructed, reuse=True)

        self._build_loss()

        self._build_optimization()

    def _build_loss(self):
        """Collect both of the losses."""
        self._build_loss_D()
        self._build_loss_G()
        self.loss_D = nameop(self.loss_D, 'loss_D')
        self.loss_G = nameop(self.loss_G, 'loss_G')
        tf.add_to_collection('losses', self.loss_D)
        tf.add_to_collection('losses', self.loss_G)

    def _build_loss_D(self):
        """Discriminator loss."""
        losses = []
        # the true examples
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_z, labels=tf.ones_like(self.D1_probs_z))))
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_z, labels=tf.ones_like(self.D2_probs_z))))
        # the generated examples
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_G, labels=tf.zeros_like(self.D1_probs_G))))
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_G, labels=tf.zeros_like(self.D2_probs_G))))
        # the reconstructed examples
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_xrecon, labels=tf.ones_like(self.D1_probs_xrecon))))
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_xrecon, labels=tf.ones_like(self.D2_probs_xrecon))))
        self.loss_D = tf.reduce_mean(losses)

    def _build_loss_G(self):
        """Generator loss."""
        losses = []
        # fool the discriminator losses
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_G, labels=tf.ones_like(self.D1_probs_G))))
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_G, labels=tf.ones_like(self.D2_probs_G))))
        # reconstruction losses
        losses.append(tf.reduce_mean((self.xb1 - self.xb1_reconstructed)**2))
        losses.append(tf.reduce_mean((self.xb2 - self.xb2_reconstructed)**2))
        # correspondences losses
        losses.append(1 * tf.reduce_mean(correspondence_loss(self.xb1, self.Gb2)))
        losses.append(1 * tf.reduce_mean(correspondence_loss(self.xb2, self.Gb1)))

        self.loss_G = tf.reduce_mean(losses)

    def _build_optimization(self):
        """Build optimization components."""
        Gvars = [tv for tv in tf.global_variables() if 'G12' in tv.name or 'G21' in tv.name]
        Dvars = [tv for tv in tf.global_variables() if 'D1' in tv.name or 'D2' in tv.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        G_update_ops = [op for op in update_ops if 'G12' in op.name or 'G21' in op.name]
        D_update_ops = [op for op in update_ops if 'D1' in op.name or 'D2' in op.name]

        with tf.control_dependencies(G_update_ops):
            optG = tf.train.AdamOptimizer(self.lr, beta1=.5, beta2=.99)
            self.train_op_G = optG.minimize(self.loss_G, var_list=Gvars, name='train_op_G')

        with tf.control_dependencies(D_update_ops):
            optD = tf.train.AdamOptimizer(self.lr, beta1=.5, beta2=.99)
            self.train_op_D = optD.minimize(self.loss_D, var_list=Dvars, name='train_op_D')

    def train(self, xb1, xb2):
        """Take a training step with batches from each domain."""
        self.iteration += 1

        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('lr:0'): self.learning_rate,
                tbn('is_training:0'): True}

        _ = self.sess.run([obn('train_op_G')], feed_dict=feed)
        _ = self.sess.run([obn('train_op_D')], feed_dict=feed)

    def get_layer(self, xb1, xb2, name):
        """Get a layer of the network by name for the entire datasets given in xb1 and xb2."""
        tensor_name = "{}:0".format(name)
        tensor = tbn(tensor_name)

        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('is_training:0'): False}

        layer = self.sess.run(tensor, feed_dict=feed)

        return layer

    def get_loss_names(self):
        """Return a string for the names of the loss values."""
        losses = [tns.name[:-2].replace('loss_', '').split('/')[-1] for tns in tf.get_collection('losses')]
        return "Losses: {}".format(' '.join(losses))

    def get_loss(self, xb1, xb2):
        """Return all of the loss values for the given input."""
        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('is_training:0'): False}

        ls = [tns for tns in tf.get_collection('losses')]
        losses = self.sess.run(ls, feed_dict=feed)

        lstring = ' '.join(['{:.3f}'.format(loss) for loss in losses])

        return lstring


class Generator(object):
    """MAGAN's generator."""

    def __init__(self,
        output_dim,
        name='',
        activation=tf.nn.relu):
        """"Initialize the generator."""
        self.output_dim = output_dim
        self.activation = activation
        self.name = name

    def __call__(self, x, reuse=False):
        """Perform the feedforward for the generator."""
        with tf.variable_scope(self.name):
            h1 = tf.layers.dense(x, 200, activation=self.activation, reuse=reuse, name='h1')
            h2 = tf.layers.dense(h1, 100, activation=self.activation, reuse=reuse, name='h2')
            h3 = tf.layers.dense(h2, 50, activation=self.activation, reuse=reuse, name='h3')

            out = tf.layers.dense(h3, self.output_dim, activation=None, reuse=reuse, name='out')

        return out

class Discriminator(object):
    """MAGAN's discriminator."""

    def __init__(self,
        name='',
        activation=tf.nn.relu):
        """Initialize the discriminator."""
        self.activation = activation
        self.name = name

    def __call__(self, x, reuse=False):
        """Perform the feedforward for the discriminator."""
        with tf.variable_scope(self.name):
            h1 = tf.layers.dense(x, 800, activation=self.activation, reuse=reuse, name='h1')
            h2 = tf.layers.dense(h1, 400, activation=self.activation, reuse=reuse, name='h2')
            h3 = tf.layers.dense(h2, 200, activation=self.activation, reuse=reuse, name='h3')
            h4 = tf.layers.dense(h3, 100, activation=self.activation, reuse=reuse, name='h4')
            h5 = tf.layers.dense(h4, 50, activation=self.activation, reuse=reuse, name='h5')

            out = tf.layers.dense(h5, 1, activation=None, reuse=reuse, name='out')

        return out


































