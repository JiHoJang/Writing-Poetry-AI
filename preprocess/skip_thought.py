import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import tensorflow as tf
import numpy as np
import json
import time
import gru_cell
import random


def initialize_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


f = open("w2i.json", 'r')
word_to_id = json.load(f)
f.close()

f = open("i2w.json", 'r')
id_to_word = json.load(f)
f.close()

encoder_dim = 500
embedding_size = 500
vocab_size = len(word_to_id)
batch_size = 32
neg_size = 64

#####################################################################
inputs = tf.placeholder(tf.int32, shape=[None])
labels = tf.placeholder(tf.int32, shape=[None])


def noise_contrastive_loss(embedding_lookup, weight_shape, bias_shape, y):
    with tf.variable_scope("nce"):
        nce_weight_init = tf.truncated_normal(weight_shape, stddev=1.0/(weight_shape[1])**0.5)
        nce_bias_init = tf.zeros(bias_shape)
        nce_W = tf.get_variable("W", initializer=nce_weight_init)
        nce_b = tf.get_variable("b", initializer=nce_bias_init)

        total_loss = tf.nn.nce_loss(weights=nce_W, biases=nce_b, inputs=embedding_lookup, labels=tf.reshape(y, [-1, 1]), num_sampled=neg_size, num_classes=vocab_size)
        return tf.reduce_mean(total_loss)


def embedding_layer(embedding_shape):
    with tf.variable_scope("embedding"):
        embedding_init = tf.random_uniform(embedding_shape, -1.0, 1.0)
        embedding_matrix = tf.get_variable(name="E", initializer=embedding_init)
        return embedding_matrix


embedding_matrix = embedding_layer([vocab_size, embedding_size])

cost2 = noise_contrastive_loss(embedding_matrix, [vocab_size, embedding_size], [vocab_size], labels)

optimizer2 = tf.train.AdamOptimizer(0.001).minimize(cost2)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "w2v_model.ckpt")
    emb_w = np.array(embedding_matrix.eval())
    sess.close()

print("Complete embedding")
#####################################################################

n_step = 20

def make_batch(seq_data):
    encode_batch, pre_batch, pos_batch = [], [], []
    mask, pre_mask, pos_mask = [], [], []

    for seq in seq_data:
        if len(seq['encode']) > n_step or len(seq['decode_pre']) > n_step or len(seq['decode_pos']) > n_step:
            continue

        for j in range(n_step - len(seq['encode'])):
            seq['encode'].append("<p>")

        for j in range(n_step - len(seq['decode_pre'])):
            seq['decode_pre'].append("<p>")

        for j in range(n_step - len(seq['decode_pos'])):
            seq['decode_pos'].append("<p>")

        temp = [word_to_id[i] for i in seq['encode']]
        temp2 = [i != "<p>" for i in seq['encode']]

        encode_batch.append(temp)
        mask.append(temp2)

        temp = [word_to_id[i] for i in seq['decode_pre']]
        temp2 = [i != "<p>" for i in seq['decode_pre']]

        pre_batch.append(temp)
        pre_mask.append(temp2)

        temp = [word_to_id[i] for i in seq['decode_pre']]
        temp2 = [i != "<p>" for i in seq['decode_pre']]

        pos_batch.append(temp)
        pos_mask.append(temp2)

    return np.asarray(encode_batch), np.asarray(pre_batch), np.asarray(pos_batch), np.asarray(mask), np.asarray(pre_mask), np.asarray(pos_mask)


f = open("skip_data.json", 'r')
poem_data = json.load(f)
f.close()
total_batch = int(len(poem_data) / batch_size)

random.shuffle(poem_data)


def random_orthonormal_initializer(shape, dtype=tf.float32, partition_info=None):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Expecting square shape, got %s" % shape)
    _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
    return u


encode_ids = tf.placeholder(tf.int32, [None, None])
decode_pre_ids = tf.placeholder(tf.int32, [None, None])
decode_post_ids = tf.placeholder(tf.int32, [None, None])

encode_mask = tf.placeholder(tf.int32, [None, None])
decode_pre_mask = tf.placeholder(tf.int32, [None, None])
decode_post_mask = tf.placeholder(tf.int32, [None, None])

uniform_initializer = tf.random_uniform_initializer(
            minval=-0.1,
            maxval=0.1)


def _initialize_gru_cell(num_units):
    return gru_cell.LayerNormGRUCell(
        num_units=num_units,
        w_initializer=uniform_initializer,
        u_initializer=random_orthonormal_initializer,
        b_initializer=tf.constant_initializer(0.0))


with tf.variable_scope("encoder") as scope:
    length = tf.to_int32(tf.reduce_sum(encode_mask, 1), name="length")

    enc_cell = _initialize_gru_cell(encoder_dim)
    _, state = tf.nn.dynamic_rnn(
        cell=enc_cell,
        inputs=tf.nn.embedding_lookup(emb_w, encode_ids),
        sequence_length=length,
        dtype=tf.float32,
        scope=scope)
    thought_vectors = tf.identity(state, name="thought_vectors")


target_cross_entropy_losses = []
target_cross_entropy_loss_weights = []

dec_prev_cell = _initialize_gru_cell(encoder_dim)
with tf.variable_scope("decoder_prev") as scope:
    embedding_prev = tf.nn.embedding_lookup(emb_w, decode_pre_ids)
    decoder_prev_input = tf.pad(
        embedding_prev[:, :-1, :], [[0, 0], [1, 0], [0, 0]], name="input")
    length = tf.reduce_sum(decode_pre_mask, 1, name="length")
    decoder_prev_output, _ = tf.nn.dynamic_rnn(
        cell=dec_prev_cell,
        inputs=decoder_prev_input,
        sequence_length=length,
        initial_state=thought_vectors,
        scope=scope)

    decoder_prev_output = tf.reshape(decoder_prev_output, [-1, encoder_dim])
    prev_targets = decode_pre_ids
    prev_targets = tf.reshape(prev_targets, [-1])
    prev_weights = tf.to_float(tf.reshape(decode_pre_mask, [-1]))

    with tf.variable_scope("logits", reuse=False) as scope:
        prev_logits = tf.contrib.layers.fully_connected(
            inputs=decoder_prev_output,
            num_outputs=vocab_size,
            activation_fn=None,
            weights_initializer=uniform_initializer,
            scope=scope)

    prev_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=prev_targets, logits=prev_logits)

    batch_prev_loss = tf.reduce_sum(prev_losses * prev_weights)
    tf.losses.add_loss(batch_prev_loss)

    tf.summary.scalar("losses/decode_pre", batch_prev_loss)

    target_cross_entropy_losses.append(prev_losses)
    target_cross_entropy_loss_weights.append(prev_weights)

dec_post_cell = _initialize_gru_cell(encoder_dim)

with tf.variable_scope("decoder_post") as scope:
    embedding_post = tf.nn.embedding_lookup(emb_w, decode_post_ids)
    decoder_post_input = tf.pad(
        embedding_post[:, :-1, :], [[0, 0], [1, 0], [0, 0]], name="input")
    length = tf.reduce_sum(decode_post_mask, 1, name="length")
    decoder_post_output, _ = tf.nn.dynamic_rnn(
        cell=dec_post_cell,
        inputs=decoder_post_input,
        sequence_length=length,
        initial_state=thought_vectors,
        scope=scope)

    decoder_post_output = tf.reshape(decoder_post_output, [-1, encoder_dim])
    post_targets = decode_post_ids
    post_targets = tf.reshape(post_targets, [-1])
    post_weights = tf.to_float(tf.reshape(decode_post_mask, [-1]))

    with tf.variable_scope("logits", reuse=False) as scope:
        post_logits = tf.contrib.layers.fully_connected(
            inputs=decoder_post_output,
            num_outputs=vocab_size,
            activation_fn=None,
            weights_initializer=uniform_initializer,
            scope=scope)

    post_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=post_targets, logits=post_logits)

    batch_post_loss = tf.reduce_sum(post_losses * post_weights)
    tf.losses.add_loss(batch_post_loss)

    tf.summary.scalar("losses/decode_post", batch_post_loss)

    target_cross_entropy_losses.append(post_losses)
    target_cross_entropy_loss_weights.append(post_weights)

total_loss = tf.losses.get_total_loss()
tf.summary.scalar("losses/total", total_loss)

global_step = tf.contrib.framework.create_global_step()

learning_rate = tf.constant(0.0008)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

sess2 = initialize_session()
sess2.run(tf.global_variables_initializer())

for epoch in range(1):
    start = time.time()

    total_cost = 0
    batch_index = 0

    for i in range(total_batch):
        batch_id, batch_pre_id, batch_pos_id, batch_mask, batch_pre_mask, batch_pos_mask = make_batch(
            poem_data[batch_index: batch_index + batch_size]
        )

        feed_dict = {encode_ids:batch_id, decode_pre_ids:batch_pre_id, decode_post_ids:batch_pos_id,
                     encode_mask:batch_pre_mask, decode_pre_mask:batch_pre_mask, decode_post_mask:batch_pos_mask}

        _, cost = sess2.run([optimizer, total_loss],
                            feed_dict=feed_dict)
        total_cost += cost

    print(time.time() - start)

    print('Epoch:', '%04d', 'cost =', '{:.6f}'.format(total_cost))

save_path = saver.save(sess2, 'skip_thought.ckpt')
