import tensorflow as tf
import numpy as np
import re
import json
import time
import os

def cleanText(readData):
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


def mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(tokens):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word

f = open("../preprocess/output2.txt", "r")
#f = open("output.txt", "r")
line = f.readline().split()
#line += '\n'

word_to_id, id_to_word = mapping(line)
f.close()

sentences = []

'''
with open("../preprocess/multim_poem.json") as json_file:
#with open("multim_poem.json") as json_file:
    data = json.load(json_file)
    for i in data:
        sentences.append(cleanText(i['poem']).split())
'''

with open("../preprocess/unim_poem.json") as json_file:
#with open("unim_poem.json") as json_file:
    data = json.load(json_file)
    for i in range(len(data)):
        sentences.append(cleanText(data[i]['poem']).split())

with open("../preprocess/gutenberg-poetry-001.ndjson") as f:
    data = ndjson.load(f)
    for p in data:
        sentences.append(cleanText(p['s']).split())


data = []

WINDOW_SIZE = 2
vocab_size = len(word_to_id)

skip_grams_inputs = []
skip_grams_labels = []

for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                #data.append([word, nb_word])
                skip_grams_inputs.append(word_to_id[word])
                skip_grams_labels.append(word_to_id[nb_word])

#print(word_to_id['is'])

batch_size = 10
total_batch = int(vocab_size / batch_size)
embedding_size = 300

tf.reset_default_graph()


inputs = tf.placeholder(tf.int32, shape=[None,])
labels = tf.placeholder(tf.int32, shape=[None,])

inputs_emb_W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
inputs_emb = tf.nn.embedding_lookup(inputs_emb_W, inputs)

W_T = tf.Variable(tf.random_uniform([embedding_size, vocab_size], -1.0, 1.0))
W_b = tf.Variable(tf.zeros([vocab_size]))
output = tf.matmul(inputs_emb, W_T) + W_b

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

session = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

init = tf.global_variables_initializer()
session.run(init)

saver = tf.train.Saver()

for epoch in range(5000):
    total_cost = 0
    start = time.time()
    for i in range(total_batch):

        batch_x = skip_grams_inputs[i*batch_size:(i+1)*batch_size]
        batch_y = skip_grams_labels[i * batch_size:(i + 1) * batch_size]

        _, loss = session.run([optimizer, cost], feed_dict = {inputs: batch_x, labels:batch_y})
        total_cost += loss

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_cost))
    print(time.time() - start)

save_path = saver.save(session, '../model/word_embedding_model.ckpt')
#save_path = saver.save(session, 'word_embedding_model.ckpt')


'''
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('
'''