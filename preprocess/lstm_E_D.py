import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import json
import time
import nltk


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


embedding_size = 300
vocab_size = len(word_to_id)
batch_size = 32
neg_size = 64

#####################################################################
inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size])

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

cost2 = noise_contrastive_loss(embedding_matrix[0:32], [vocab_size, embedding_size], [vocab_size], labels)

optimizer2 = tf.train.AdamOptimizer(0.001).minimize(cost2)
#####################################################################

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "w2v_model.ckpt")
    emb_w = np.array(embedding_matrix.eval())
    sess.close()

print("Complete embedding")

def cosine_similiarty(x, y):
    return np.sum(x * y) / (np.sum(x**2)**0.5 * np.sum(y**2)**0.5)


def similar_words(word="good"):
    idx = word_to_id[word]

    scores = []
    for i in range(len(emb_w)):
        cosine_sim = cosine_similiarty(emb_w[i], emb_w[idx])
        scores.append((cosine_sim, i))
    scores = sorted(scores, reverse=True)

    ret = []

    ret.append(scores[0][1])
    ret.append(scores[1][1])
    ret.append(scores[2][1])

    return ret


n_step = 15
n_hidden = 128
n_class = len(word_to_id)
#n_class = 10000

tf.reset_default_graph()

enc_input = tf.placeholder(tf.int32, [None, None])
dec_input = tf.placeholder(tf.int32, [None, None])
targets = tf.placeholder(tf.int32, [None, None])
_labels = tf.nn.embedding_lookup(emb_w, targets)

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    enc_emb = tf.nn.embedding_lookup(emb_w, enc_input)
    _, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_emb, dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    dec_emb = tf.nn.embedding_lookup(emb_w, dec_input)
    outputs, _ = tf.nn.dynamic_rnn(dec_cell, dec_emb, initial_state=enc_states, dtype=tf.float32)

model = tf.layers.dense(outputs, embedding_size, activation=None)


#cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=tf.nn.embedding_lookup(emb_w, targets)))
cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=_labels, predictions=model))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


def make_batch(seq_data):
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        isNoun = lambda pos: pos[:2] == 'NN'

        tokenized = nltk.word_tokenize(seq)

        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if isNoun(pos)]

        seq = seq.split()

        if len(nouns) == 0:
            print("Not nouns")
            continue
        else:
            input = [n for n in similar_words(nouns[0])]
            for i in range(n_step - len(seq)):
                seq.append("<eos>")
            
            seq_output = seq.copy()
            seq_output.insert(0, '<eos>')
            output = [word_to_id[n] for n in seq_output]
            seq_target = seq.copy()
            seq_target.append("<eos>")
            target = [word_to_id[n] for n in seq_target]

            input_batch.append(input)
            output_batch.append(output)
            target_batch.append(target)

    return input_batch, output_batch, target_batch


batch_size = 2
total_batch = int(len(sentences) / batch_size)

sess2 = initialize_session()
sess2.run(tf.global_variables_initializer())


for epoch in range(1):
    start = time.time()

    total_loss = 0

    batch_index = 0

    for i in range(2):
        input_batch, output_batch, target_batch = make_batch(sentences[batch_index:batch_index+batch_size])
        batch_index += batch_size
        print('training')
        _, loss = sess2.run([optimizer, cost], feed_dict={enc_input:input_batch, dec_input:output_batch, targets:target_batch})
        total_loss += loss

    print('Epoch:', '%04d', 'cost =', '{:.6f}'.format(total_loss))
    print(time.time() - start)


def generation(word='sun'):
    input_batch = [n for n in similar_words(word)]
    output_batch = [word_to_id["<eos>"] for i in range(n_step)]

    result = sess2.run(model, feed_dict={enc_input: [input_batch], dec_input: [output_batch]})

    decoded = []
    for i in result[0]:

        scores = []
        for j in range(len(emb_w)):
            cosine_sim = cosine_similiarty(emb_w[j], i)
            scores.append((cosine_sim, j))
        scores = sorted(scores, reverse=True)

        wordvec = similar_words(id_to_word[str(scores[0][1])])
        print(wordvec)
        decoded.append(wordvec[0])

    print(decoded)

generation()

sess2.close()


