import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import re
import json
import ndjson
import time
from textblob import TextBlob
from langdetect import detect


def cleanText(readData) :
    string = readData.replace('\"', '').replace('_', ' ').lower()
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`_]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string


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

f = open("sentences.txt", 'r')
for x in f:
    sentences.append(x)
f.close()


sess = tf.Session()
saver = tf.train.import_meta_graph('w2v_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
embedding_matrix = graph.get_tensor_by_name("E:0")

emb_w = np.array(sess.run(embedding_matrix))
sess.close()


def similar_words(word="good"):
    idx = word_to_id[word]

    scores = []
    for i in range(len(emb_w)):
        cosine_sim = cosine+similiarty(emb_w[i], emb_w[idx])
        scores.append((cosine_sim, i))
    scores = sorted(scores, reverse=True)

    return scores[0:2]


n_step = 15
n_hidden = 128
n_class = len(word_to_id)

enc_input = tf.placeholder(tf.float32, [None, None, n_class])
dec_input = tf.placeholder(tf.float32, [None, None, n_class])
targets = tf.placeholder(tf.int64, [None, None])

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    _, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    outputs. _ = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)

model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


def make_match(seq_data):
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        blob = TextBlob(seq)

        if len(blob.noun_phrases) == 0:
            continue
        else:
            input = [n for n in similar_words(blob.noun_phrases[0])]

            for i in range(n_step - len(seq)):
                seq.append("<eos>")
            
            seq_output = seq.copy()
            seq_output.insert(0, '<eos>')
            output = [emb_w[word_to_id[n]] for n in seq_output]
            seq_target = seq.copy()
            seq_target.append("<eos>")
            target = [emb_w[word_to_id[n]] for n in seq_target]

            input_batch.append(np.eye(n_class)[input])
            output_batch.append(np.eye(n_class)[output])
            target_batch.append(target)

    return input_batch, output_batch, target_batch


sess2 = initialize_session()
sess2.run(tf.global_variables_initializer())
input_batch, ouput_batch, target_batch = make_match(sentences)

for epoch in range(5000):
    start = time.time()
    _, loss = sess2.run([optimizer, cost], feed_dict={enc_input: input_batch, dec_input: output_batch, targets: target_batch})

    print('Epoch:', '%04d', 'cost =', '{:.6f}'.format(loss))
    print(time.time() - start)


def generation(word = 'sun'):
    input_batch, output_batch, _ = make_batch([word])
    prediction = tf.argmax(model, 2)

    result = sess2.run(prediction, feed_dict={enc_input: input_batch, dec_input: output_batch})
    decoded = []
    for i in result[0]:
        wordvec = similar_words(i)[0]
        decoded += id_to_word(str(np.where(emb_w == wordvec)))

    print(decoded)

sess2.close()


