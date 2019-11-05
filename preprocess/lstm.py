from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import json
import time
import nltk
import random
#import input_ops


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

sentences = []


'''
f = open("sentences.txt", 'r')
for x in f:
    isNoun = lambda pos: pos[:2] == 'NN'

    tokenized = nltk.word_tokenize(x)

    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if isNoun(pos)]
    if len(nouns) > 0:
        sentences.append(x)
f.close()

f = open("sentences2.txt", 'w')
for i in sentences:
    f.write(i)
    f.write('\n')
f.close()

print("complete sentences")
'''

f = open("sentences.txt", 'r')
for x in f:
    sentences.append(x)
f.close()

random.shuffle(sentences)

embedding_size = 500
vocab_size = len(word_to_id)
batch_size = 32
neg_size = 64

#####################################################################
inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size])


def noise_contrastive_loss(embedding_lookup, weight_shape, bias_shape, y):
    with tf.variable_scope("nce"):
        nce_weight_init = tf.truncated_normal(weight_shape, stddev=1.0 / (weight_shape[1]) ** 0.5)
        nce_bias_init = tf.zeros(bias_shape)
        nce_W = tf.get_variable("W", initializer=nce_weight_init)
        nce_b = tf.get_variable("b", initializer=nce_bias_init)

        total_loss = tf.nn.nce_loss(weights=nce_W, biases=nce_b, inputs=embedding_lookup, labels=tf.reshape(y, [-1, 1]),
                                    num_sampled=neg_size, num_classes=vocab_size)
        return tf.reduce_mean(total_loss)


def embedding_layer(embedding_shape):
    with tf.variable_scope("embedding"):
        embedding_init = tf.random_uniform(embedding_shape, -1.0, 1.0)
        embedding_matrix = tf.get_variable(name="E", initializer=embedding_init)
        return embedding_matrix


embedding_matrix = embedding_layer([vocab_size, embedding_size])

cost2 = noise_contrastive_loss(embedding_matrix[0:32], [vocab_size, embedding_size], [vocab_size], labels)

optimizer2 = tf.train.AdamOptimizer(0.001).minimize(cost2)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "w2v_model.ckpt")
    emb_w = np.array(embedding_matrix.eval())
    sess.close()

print("Complete embedding")
#####################################################################


def cosine_similarty(x, y):
    return np.sum(x * y) / (np.sum(x ** 2) ** 0.5 * np.sum(y ** 2) ** 0.5)


def similar_words(word="good"):
    idx = word_to_id[word]

    scores = []
    for i in range(len(emb_w)):
        cosine_sim = cosine_similarty(emb_w[i], emb_w[idx])
        scores.append((cosine_sim, i))
    scores = sorted(scores, reverse=True)

    ret = []

    ret.append(scores[0][1])
    ret.append(scores[1][1])
    ret.append(scores[2][1])

    return ret


n_step = 21
n_hidden = 128
n_class = len(word_to_id)
num_layers = 3

tf.reset_default_graph()


init_scale = 1.0

uniform_initializer = tf.random_uniform_initializer(minval=-init_scale, maxval=init_scale)

enc_input = tf.placeholder(tf.int32, [None, None])
dec_input = tf.placeholder(tf.int32, [None, None])
target_mask = tf.placeholder(tf.float32, [None, None])
targets = tf.placeholder(tf.int64, [None, None])

#_labels = tf.nn.embedding_lookup(emb_w, targets)

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    #enc_cell = tf.nn.rnn_cell.MultiRNNCell([enc_cell] * num_layers)
    enc_emb = tf.nn.embedding_lookup(emb_w, enc_input)
    _, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_emb, dtype=tf.float32)

with tf.variable_scope('decode') as scope1:
    dec_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    #dec_cell = tf.nn.rnn_cell.MultiRNNCell([dec_cell] * num_layers)
    dec_emb = tf.nn.embedding_lookup(emb_w, dec_input)
    #dec_emb_input = tf.pad(dec_emb[:, :-1, :], [[0, 0], [1, 0], [0, 0]], name="input")
    length = tf.reduce_sum(target_mask, 1, name="length")
    outputs, dec_state = tf.nn.dynamic_rnn(
        cell=dec_cell,
        inputs=dec_emb,
        initial_state=enc_states,
        dtype=tf.float32,
        #sequence_length=length,
        scope=scope1
    )
    '''
    outputs = tf.reshape(outputs, [-1, n_hidden])
    print(outputs)
    targets = tf.reshape(targets, [-1])
    weights = tf.to_float(tf.reshape(target_mask, [-1]))
    '''

with tf.variable_scope("logits") as scope2:
    model = tf.contrib.layers.fully_connected(
        inputs=outputs,
        num_outputs=vocab_size,
        activation_fn=None,
        weights_initializer=uniform_initializer,
        scope=scope2)

# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=tf.nn.embedding_lookup(emb_w, targets)))
cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=model)
loss = tf.reduce_sum(cost)
optimizer = tf.train.AdamOptimizer(0.0008).minimize(loss)


def make_batch(seq_data):
    input_batch, output_batch, target_batch = [], [], []
    target_mask = []


    for seq in seq_data:
        isNoun = lambda pos: pos[:2] == 'NN'

        tokenized = nltk.word_tokenize(seq)

        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if isNoun(pos)]

        seq = seq.split()

        if len(nouns) == 0 or len(seq) >= n_step:
            #print("Not nouns")
            continue
        elif nouns[0] in word_to_id:
            input = [word_to_id[nouns[0]]]
            for i in range(n_step - len(seq)):
                seq.append("<p>")

            seq_output = seq.copy()
            seq_output.insert(0, '<go>')
            output = [word_to_id[n] for n in seq_output]
            seq_target = seq.copy()
            seq_target.append("<p>")
            target = [word_to_id[n] for n in seq_target]
            mask = []
            for i in target:
                if i == word_to_id["<p>"]:
                    mask.append(0)
                else:
                    mask.append(1)

            input_batch.append(np.asarray(input))
            output_batch.append(np.asarray(output))
            target_batch.append(np.array(target))
            target_mask.append(np.array(mask))
    return input_batch, output_batch, target_batch, target_mask


batch_size = 128
total_batch = int(len(sentences) / batch_size)

sess2 = initialize_session()
sess2.run(tf.global_variables_initializer())

saver = tf.train.Saver()

for epoch in range(50):
    start = time.time()

    total_loss = 0

    batch_index = 0

    for i in range(2000):

        input_batch, output_batch, target_batch, mask_batch = make_batch(sentences[batch_index:batch_index + batch_size])
        batch_index += batch_size

        _, cost2 = sess2.run([optimizer, loss],
                            feed_dict={enc_input: input_batch, dec_input: output_batch, targets: target_batch,
                                       target_mask: mask_batch})
        total_loss += cost2

    print(time.time() - start)

    print('Epoch:', '%04d' % epoch, 'cost =', '{:.6f}'.format(total_loss))

save_path = saver.save(sess2, 'lstmED.ckpt')

'''
def generation(word='sun'):
    #input_batch = [n for n in similar_words(word)]
    input_batch = [word_to_id[word]]
    output_batch = []
    output_batch.append(word_to_id["<go>"])
    prediction = tf.arg_max(model, 2)

    result, enc = sess2.run([prediction, enc_states], feed_dict={enc_input: [input_batch], dec_input: [output_batch]})

    decoded = []
    decoded.append(id_to_word[str(result[0, -1])])
    output_batch.append(result[0, -1])
    print(result)

    for i in range(n_step):
        feed_dict={dec_input: [output_batch], enc_states: enc}
        result, enc = sess2.run([prediction, dec_state], feed_dict=feed_dict)
        decoded.append(id_to_word[str(result[0, -1])])
        output_batch.append(result[0, -1])
        print(result)

    for i in range(n_step):
        prediction = tf.arg_max(model, 2)

        result = sess2.run(prediction, feed_dict={enc_input: [input_batch], dec_input: [output_batch]})
        print(result)
        decoded.append(id_to_word[str(result[0, i])])
        output_batch[i+1] = int(result[0, i])

    for i in result[0]:
        scores = []
        for j in range(len(emb_w)):
            cosine_sim = cosine_similarty(emb_w[j], i)
            scores.append((cosine_sim, j))
        scores = sorted(scores, reverse=True)
`       
        wordvec = similar_words(id_to_word[str(scores[0][1])])
        print(wordvec)
        decoded.append(wordvec[0])
        decoded.append(id_to_word[str(i)])

    print(decoded)


generation()
                '''

sess2.close()
