import tensorflow as tf
import numpy as np
import simplejson
import threading
import utils.train_utils as train_utils
import utils.googlenet_load as googlenet_load
import tensorflow.contrib.rnn as rnn
import time
import tensorflow.contrib.slim as slim
import string
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import argparse

@ops.RegisterGradient('Hungarian')
def _hungarian_grad(op, *args):
    return map(array_ops.zeros_like, op.inputs)

def enqueue_thread(sess, coord, enqueue, gen, phase, x_in, box_in, conf_in):
    while not coord.should_stop():
        data_gen = gen.next()
        image = data_gen['image']
        confs = data_gen['confs']
        boxes = data_gen['boxes']
        sess.run(enqueue, feed_dict={x_in[phase]:image, box_in[phase]:boxes, conf_in[phase]:confs})


def build_lstm(x, H, reuse):
    '''
    Building LSTM layers
    '''
    rnn_len = H['rnn_len']
    num_states = H['num_states']
    num_lstm_layers = H['num_lstm_layers']
    grid_height = H['grid_height']
    grid_width = H['grid_width']
    grid_size = grid_height * grid_width
    batch_size = H['batch_size']
    outer_size = batch_size * grid_size

    lstm_cells = rnn.MultiRNNCell(
        cells=[rnn.BasicLSTMCell(num_units=num_states, reuse=reuse) for _ in range(num_lstm_layers)])

    state = lstm_cells.zero_state(batch_size=outer_size, dtype=tf.float32)

    lstm_outputs = []

    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('lstm', reuse=reuse, initializer=initializer):
        for i in range(rnn_len):
            if i > 0: tf.get_variable_scope().reuse_variables()
            lstm_out, state = lstm_cells(x, state)
            lstm_outputs.append(lstm_out)
    return lstm_outputs


def build_pred(H, x_in, phase, reuse):
    '''
    Building the feedforward prediction NN for both training and test
    '''
    grid_height = H['grid_height']
    grid_width = H['grid_width']
    grid_size = grid_height * grid_width
    batch_size = H['batch_size']
    outer_size = batch_size * grid_size
    later_feat_channels = H['later_feat_channels']
    num_states = H['num_states']
    rnn_len = H['rnn_len']

    is_training = {'train': True, 'test': False}[phase]

    x_in -= 117
    feat_5c, feat_3b = googlenet_load.model(x=x_in, is_training=is_training, H=H, reuse=reuse)

    scaling_factor = 0.01
    lstm_input = tf.reshape(feat_5c * scaling_factor, shape=[outer_size, later_feat_channels])

    lstm_outputs = build_lstm(lstm_input, H, reuse)

    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    pred_boxes_c, pred_logits_c = [], []
    with tf.variable_scope('decoder', reuse=reuse, initializer=initializer):
        for i in range(rnn_len):
            box_weight = tf.get_variable(dtype=tf.float32, name='box_weight_%i' % i, shape=[num_states, 4])
            logit_weight = tf.get_variable(dtype=tf.float32, name='conf_weight_%i' % i, shape=[num_states, 2])
            pred_box = tf.matmul(lstm_outputs[i], box_weight) * 50.
            pred_logit = tf.matmul(lstm_outputs[i], logit_weight)
            pred_box_r = tf.reshape(pred_box, shape=[outer_size, 1, 4])
            pred_logit_r = tf.reshape(pred_logit, shape=[outer_size, 1, 2])
            pred_boxes_c.append(pred_box_r)
            pred_logits_c.append(pred_logit_r)
        pred_boxes = tf.concat(values=pred_boxes_c, axis=1)
        pred_logits = tf.concat(values=pred_logits_c, axis=1)
        pred_confs = tf.nn.softmax(logits=pred_logits)

    return pred_boxes, pred_logits, pred_confs


def build_loss(H, q):
    '''
    Building the loss and accuracy
    '''
    grid_height = H['grid_height']
    grid_width = H['grid_width']
    grid_size = grid_height * grid_width
    batch_size = H['batch_size']
    outer_size = batch_size * grid_size
    rnn_len = H['rnn_len']
    hungarian_iou = H['hungarian_iou']
    conf_weight = H['conf_weight']
    box_weight = H['box_weight']

    pred_boxes, pred_logits, pred_confs, loss_box, loss_conf, accuracy, loss = {}, {}, {}, {}, {}, {}, {}
    hungarian_module = tf.load_op_library('utils/hungarian/hungarian.so')

    for phase in ['train', 'test']:
        x_in, box_in, conf_in = q[phase].dequeue_many(batch_size)

        flag_in = tf.arg_max(conf_in, dimension=-1)

        box_r = tf.reshape(box_in, shape=[outer_size, rnn_len, 4])
        flag_r = tf.cast(tf.reshape(flag_in, shape=[outer_size, rnn_len]), tf.int32)

        reuse = {'train': False, 'test': True}[phase]
        pred_boxes[phase], pred_logits[phase], pred_confs[phase] = build_pred(H, x_in, phase, reuse)
        # classes is the reordered correct label
        # perm_truth is the reordered correct boxes
        # pred_mask is the mask for the predicted boxes
        assignments, classes, perm_truth, pred_mask = hungarian_module.hungarian(
            pred_boxes[phase], box_r, flag_r, hungarian_iou)
        true_class = tf.cast(tf.greater(classes, 0), dtype=tf.int32)
        loss_conf[phase] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred_logits[phase], labels=true_class)) * conf_weight
        residual = tf.abs(perm_truth - pred_boxes[phase] * pred_mask)
        loss_box[phase] = tf.reduce_mean(residual) * box_weight
        loss[phase] = loss_box[phase] + loss_conf[phase]
        pred_flag = tf.cast(tf.arg_max(pred_logits[phase], dimension=-1), tf.int32)
        accuracy[phase] = tf.reduce_mean(tf.cast(tf.equal(flag_r, pred_flag), tf.float32))
    return pred_boxes, pred_confs, loss, loss_box, loss_conf, accuracy


def build(H, q):
    '''
    Building the train_op, summary
    '''
    clip_norm = H['clip_norm']
    log_dir = 'logdir'
    pred_boxes, pred_confs, loss, loss_box, loss_conf, accuracy = build_loss(H, q)

    # train_op
    lr = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.RMSPropOptimizer(decay=0.9, learning_rate=lr, epsilon=0.0001)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss['train'], tvars)
    grads, norm = tf.clip_by_global_norm(grads, clip_norm)
    global_step = tf.Variable(0, trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)

    # building the summary
    for phase in ['train', 'test']:
        tf.summary.scalar('%s/loss' % phase, loss[phase])
        tf.summary.scalar('%s/accuracy' % phase, accuracy[phase])

    summary_op = tf.summary.merge_all()
    return train_op, summary_op, global_step, loss, accuracy, lr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes', required=True, type=str)

    args = parser.parse_args()
    hypes = args.hypes

    with open(hypes) as f:
        H = simplejson.load(f)
    image_height = H['image_height']
    image_width = H['image_width']
    queue_size = H['queue_size']
    rnn_len = H['rnn_len']
    num_classes = H['num_classes']
    grid_width = H['grid_width']
    grid_height = H['grid_height']
    grid_size = grid_height * grid_width
    batch_size = H['batch_size']
    clip_norm = H['clip_norm']
    max_iter = H['max_iter']
    display_iter = H['display_iter']
    learning_rate = H['learning_rate']
    lr_step = H['lr_step']
    save_iter = H['save_iter']
    log_dir = H['log_dir']

    queue, enqueue_op, x_in, box_in, conf_in = {}, {}, {}, {}, {}
    shape = ([image_height, image_width, 3],
             [grid_size, rnn_len, 4],
             [grid_size, rnn_len, num_classes]
             )
    for phase in ['train', 'test']:
        x_in[phase] = tf.placeholder(tf.float32)
        box_in[phase] = tf.placeholder(tf.float32)
        conf_in[phase] = tf.placeholder(tf.float32)
        queue[phase] = tf.FIFOQueue(capacity=queue_size,
                                    shapes=shape,
                                    dtypes=(tf.float32, tf.float32, tf.float32)
                                    )
        enqueue_op[phase] = queue[phase].enqueue((x_in[phase], box_in[phase], conf_in[phase]))

    train_op, summary_op, global_step, loss, accuracy, lr = build(H, queue)

    summary_writer = tf.summary.FileWriter(logdir=log_dir + '/summary/', flush_secs=10)
    saver = tf.train.Saver(max_to_keep=None)

    with tf.Session() as sess:
        t, gen = {}, {}
        coord = tf.train.Coordinator()
        for phase in ['train', 'test']:
            gen[phase] = train_utils.load_data_gen(H=H, jitter=False, phase=phase)
            t[phase] = threading.Thread(target=enqueue_thread,
                                        args=(sess, coord, enqueue_op[phase], gen[phase], phase, x_in, box_in, conf_in))
            t[phase].start()
        summary_writer.add_graph(sess.graph)
        start = time.time()
        sess.run(tf.global_variables_initializer())
        init_fn = slim.assign_from_checkpoint_fn(
            ignore_missing_vars=True,
            model_path='data/inception_v1.ckpt',
            var_list=slim.get_model_variables('InceptionV1'))
        init_fn(sess)
        for i in range(max_iter):
            adjusted_lr = learning_rate * 0.5 ** max(0, (np.int32(i / lr_step) - 2))
            if i % display_iter != 0:
                sess.run([train_op, loss['train']], feed_dict={lr: adjusted_lr})
            else:
                if i > 0:
                    dt = (time.time() - start) / batch_size / display_iter
                start = time.time()
                train_loss, test_accuracy, summaries, _ = sess.run(
                    [loss['train'], accuracy['test'], summary_op, train_op], feed_dict={lr: adjusted_lr})
                summary_writer.add_summary(summaries, global_step=global_step.eval())
                print_str = string.join([
                    'Step: %d',
                    'lr: %f',
                    'Train loss: %.2f',
                    'Test accuracy: %.1f',
                    'Time/image (ms) %.1f'
                ])
                print(print_str % (i, adjusted_lr, train_loss, test_accuracy * 100, dt * 1000 if i > 0 else 0))
            if global_step.eval() % save_iter == 0 or global_step.eval() == max_iter - 1:
                saver.save(sess, log_dir + '/checkpoint/', global_step=global_step.eval())
    coord.request_stop()
    coord.join()
if __name__ == '__main__':
    main()