import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import tensorflow as tf
import numpy as np
import json
import time
import gru_cell
import random
import copy


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

optimizer2 = tf.train.AdamOptimizer(0.0008).minimize(cost2)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "/ckpt/w2v_model.ckpt")
    emb_w = np.array(embedding_matrix.eval())
    sess.close()

tf.reset_default_graph()

print("Complete embedding")
#####################################################################

n_step = 20

def make_batch(seq_data):
    encode_batch, pre_batch, pos_batch = [], [], []
    mask, pre_mask, pos_mask = [], [], []
    target_pre, target_pos = [], []

    for seq2 in seq_data:
        seq = copy.deepcopy(seq2)

        if len(seq['encode']) >= (n_step-1) or len(seq['decode_pre']) >= (n_step-1) or len(seq['decode_pos']) >= (n_step-1):
            print("too long!!!")
            continue

        try:

            for j in range(n_step - len(seq['encode'])):
                seq['encode'].append("<p>")

            #seq['decode_pre'].insert(0, '<go>')
            for j in range(n_step - len(seq['decode_pre'])-1):
                seq['decode_pre'].append("<p>")

            #seq['decode_pos'].insert(0, '<go>')
            for j in range(n_step - len(seq['decode_pos'])-1):
                seq['decode_pos'].append("<p>")

            temp1 = [word_to_id[i] for i in seq['encode']]
            temp2 = [i != "<p>" for i in seq['encode']]

            temp3 = [word_to_id[i] for i in seq['decode_pre']]
            #temp4 = [i != "<p>" for i in seq['decode_pre']]

            temp5 = [word_to_id[i] for i in seq['decode_pre']]
            #temp6 = [i != "<p>" for i in seq['decode_pre']]

            temp7 = copy.deepcopy(temp3)
            temp8 = copy.deepcopy(temp5)
            temp3.insert(0, word_to_id['<go>'])
            temp5.insert(0, word_to_id['<go>'])

            temp7.append(word_to_id['<p>'])
            temp8.append(word_to_id['<p>'])

            temp4 = [i != word_to_id["<p>"] for i in temp7]
            temp6 = [i != word_to_id["<p>"] for i in temp8]

            encode_batch.append(temp1)
            mask.append(temp2)
            pre_batch.append(temp3)
            pre_mask.append(temp4)
            pos_batch.append(temp5)
            pos_mask.append(temp6)
            target_pre.append(temp7)
            target_pos.append(temp8)

        except Exception:
            pass

    return np.asarray(encode_batch), np.asarray(pre_batch), np.asarray(pos_batch), np.asarray(mask), np.asarray(pre_mask), np.asarray(pos_mask), np.asarray(target_pre), np.array(target_pos)

poem_data=[]
f = open("skip_data.json", 'r')
t = json.load(f)
for x in t:
    if len(x['encode']) < (n_step-1) and len(x['decode_pre']) < (n_step-1) and len(x['decode_pos']) < (n_step-1):

        poem_data.append(x)
f.close()
total_batch = int(len(poem_data) / batch_size)
print(total_batch)

random.shuffle(poem_data)


def random_orthonormal_initializer(shape, dtype=tf.float32, partition_info=None):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Expecting square shape, got %s" % shape)
    _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
    return u


encode_ids = tf.placeholder(tf.int32, [None, None])
decode_pre_ids = tf.placeholder(tf.int32, [None, None])
decode_post_ids = tf.placeholder(tf.int32, [None, None])

encode_mask = tf.placeholder(tf.float32, [None, None])
decode_pre_mask = tf.placeholder(tf.float32, [None, None])
decode_post_mask = tf.placeholder(tf.float32, [None, None])

prev_targets = tf.placeholder(tf.int32, [None, None])
post_targets = tf.placeholder(tf.int32, [None, None])

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
    #decoder_prev_input = tf.pad(embedding_prev[:, :-1, :], [[0, 0], [1, 0], [0, 0]], name="input")
    length = tf.reduce_sum(decode_pre_mask, 1, name="length")
    decoder_prev_output, _ = tf.nn.dynamic_rnn(
        cell=dec_prev_cell,
        inputs=embedding_prev,
        #sequence_length=length,
        initial_state=thought_vectors,
        scope=scope)
    '''
    decoder_prev_output = tf.reshape(decoder_prev_output, [-1, encoder_dim])
    prev_targets = decode_pre_ids
    prev_targets = tf.reshape(prev_targets, [-1])
    prev_weights = tf.to_float(tf.reshape(decode_pre_mask, [-1]))
    '''

    with tf.variable_scope("logits", reuse=False) as scope:
        prev_logits = tf.contrib.layers.fully_connected(
            inputs=decoder_prev_output,
            num_outputs=vocab_size,
            activation_fn=None,
            weights_initializer=uniform_initializer,
            scope=scope)

    prev_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=prev_targets, logits=prev_logits)

    batch_prev_loss = tf.reduce_sum(prev_losses * decode_pre_mask)
    tf.losses.add_loss(batch_prev_loss)

    tf.summary.scalar("losses/decode_pre", batch_prev_loss)

    target_cross_entropy_losses.append(prev_losses)
    #target_cross_entropy_loss_weights.append(prev_weights)

dec_post_cell = _initialize_gru_cell(encoder_dim)

with tf.variable_scope("decoder_post") as scope:
    embedding_post = tf.nn.embedding_lookup(emb_w, decode_post_ids)
    #decoder_post_input = tf.pad(embedding_post[:, :-1, :], [[0, 0], [1, 0], [0, 0]], name="input")
    length = tf.reduce_sum(decode_post_mask, 1, name="length")
    decoder_post_output, _ = tf.nn.dynamic_rnn(
        cell=dec_post_cell,
        inputs=embedding_post,
        #sequence_length=length,
        initial_state=thought_vectors,
        scope=scope)
    '''
    decoder_post_output = tf.reshape(decoder_post_output, [-1, encoder_dim])
    post_targets = decode_post_ids
    post_targets = tf.reshape(post_targets, [-1])
    post_weights = decode_post_mask
    '''
    with tf.variable_scope("logits", reuse=False) as scope:
        post_logits = tf.contrib.layers.fully_connected(
            inputs=decoder_post_output,
            num_outputs=vocab_size,
            activation_fn=None,
            weights_initializer=uniform_initializer,
            scope=scope)

    post_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=post_targets, logits=post_logits)

    batch_post_loss = tf.reduce_sum(post_losses * decode_post_mask)
    tf.losses.add_loss(batch_post_loss)

    tf.summary.scalar("losses/decode_post", batch_post_loss)

    target_cross_entropy_losses.append(post_losses)
    #target_cross_entropy_loss_weights.append(post_weights)

total_loss = tf.losses.get_total_loss()
tf.summary.scalar("losses/total", total_loss)

global_step = tf.contrib.framework.create_global_step()

learning_rate = tf.constant(0.001)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

saver2 = tf.train.Saver()

sess2 = initialize_session()
sess2.run(tf.global_variables_initializer())

batch_id, batch_pre_id, batch_pos_id, batch_mask, batch_pre_mask, batch_pos_mask = [], [], [], [], [], []
batch_pos, batch_pre = [], []

for epoch in range(40):
    start = time.time()

    total_cost = 0
    batch_index = 0

    index = 0

    for i in range(1800):
        #print("training %d " % index)
        index += 1

        batch_id, batch_pre_id, batch_pos_id, batch_mask, batch_pre_mask, batch_pos_mask, batch_pre, batch_pos = make_batch(
            poem_data[batch_index: batch_index + batch_size])
        batch_index += batch_size

        feed_dict = {encode_ids:batch_id, decode_pre_ids:batch_pre_id, decode_post_ids:batch_pos_id,
                     encode_mask:batch_pre_mask, decode_pre_mask:batch_pre_mask, decode_post_mask:batch_pos_mask,
                     prev_targets:batch_pre, post_targets:batch_pos}
        #if len(batch_id) == batch_size and len(batch_pre_id) == batch_size and len(batch_pos_id) == batch_size and len(batch_mask) == batch_size and len(batch_pre_mask) == batch_size and len(batch_pos_mask) == batch_size:
        _, cost = sess2.run([optimizer, total_loss],
                            feed_dict=feed_dict)

        total_cost += cost

    print(time.time() - start)

    print('Epoch:', '%04d' % epoch, 'cost =', '{:.6f}'.format(total_cost))


save_path = saver2.save(sess2, '/ckpt/skip_thought.ckpt')


'''
def generation(sentence=['the', 'sunny', 'street']):
    test = dict()
    test['encode'] = sentence
    test['decode_pre'] = ["<p>"]
    test['decode_pos'] = ["<p>"]

    test_encode, test_pre, test_pos, test_mask, test_pre_mask, test_pos_mask, _, _ = make_batch([test])

    prediction = tf.arg_max(post_logits, 2)

    feed_dict = {encode_ids:test_encode, decode_pre_ids:test_pre, decode_post_ids:test_pos,
                 encode_mask:test_mask, decode_pre_mask:test_pre_mask, decode_post_mask:test_pos_mask}

    result, thoughts = sess2.run([prediction, thought_vectors], feed_dict=feed_dict)
    print(result)

    #print(test_pos)

    decode = []
    test_pos[0, 0] = result[0, 0]
    temp = result[0, 0] != word_to_id["<p>"]
    test_pos_mask[0, 0] = temp
    decode.append(id_to_word[str(result[0, 0])])

    for i in range(1, n_step):
        feed_dict = {thought_vectors: thoughts, decode_post_ids:test_pos, decode_post_mask:test_pos_mask}
        result = sess2.run(prediction, feed_dict=feed_dict)
        test_pos[0, i] = result[0, i]
        temp = result[0, i] != word_to_id["<p>"]
        test_pos_mask[0, i] = temp
        decode.append(id_to_word[str(result[0, i])])
        #print(test_pos)

    print(decode)
generation()


'''