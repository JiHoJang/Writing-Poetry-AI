import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import re
import json
import ndjson
import time
from langdetect import detect
import os

words = []

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
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", "", string)
    return string


def mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(tokens):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word

def initialize_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


sentences = []


with open("unim_poem.json") as json_file:
    data = json.load(json_file)
    for i in range(len(data)):
        try:
            if detect(data[i]['poem']) == 'en':
                sen = data[i]['poem'].split('\n')
                for s in sen:
                    s = cleanText(s)
                    sentences.append(s)
                    words += s.split()
        except Exception:
            pass

'''
with open("unim_poem.json") as json_file:
    data = json.load(json_file)
    for i in range(len(data)):
        sen = data[i]['poem'].split('\n')
        for s in sen:
            s = cleanText(s)
            sentences.append(s)
            words += s.split()
'''


print(len(sentences))
with open("gutenberg-poetry-v001.ndjson") as f:
    data = ndjson.load(f)
    for i in range(100000):
        temp = cleanText(data[i]['s'])
        sentences.append(temp)
        words += temp.split()

print("Complete open gutenberg")

words = list(set(words))

print(len(words))

f = open("output2.txt", 'w')
for i in words:
    f.write(i + ' ')
f.write('<eos>')
f.close()

f = open("output2.txt", "r")
#f = open("output.txt", "r")
line = f.readline().split()
#line += '\n'

f = open("sentences.txt", "w")
for i in sentences:
    for j in i:
        f.write(j + ' ')
    f.write('\n')
f.close()

word_to_id, id_to_word = mapping(line)
f.close()

json1 = json.dumps(word_to_id)
f=open("w2i.json", "w")
f.write(json1)
f.close()

json2 = json.dumps(id_to_word)
f=open("i2w.json", "w")
f.write(json2)
f.close()

data = []

WINDOW_SIZE = 4
vocab_size = len(word_to_id)
neg_size = 64

skip_grams_inputs = []
skip_grams_labels = []

inputs_labels = []

for sentence in sentences:
    sentence = sentence.split()
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                inputs_labels.append((word_to_id[word], word_to_id[nb_word]))


#print(word_to_id['is'])
print("Complete make sentences")


batch_size = 32
total_batch = int(vocab_size / batch_size)
embedding_size = 300

tf.reset_default_graph()


inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size])

#weight_shape = [vocab_size, embedding_size]


def noise_contrastive_loss(embedding_lookup, weight_shape, bias_shape, y):
    with tf.variable_scope("nce"):
        nce_weight_init = tf.truncated_normal(weight_shape, stddev=1.0/(weight_shape[1])**0.5)
        nce_bias_init = tf.zeros(bias_shape)
        nce_W = tf.get_variable("W", initializer=nce_weight_init)
        nce_b = tf.get_variable("b", initializer=nce_bias_init)

        total_loss = tf.nn.nce_loss(weights=nce_W, biases=nce_b, inputs=embedding_lookup, labels=tf.reshape(y, [-1, 1]), num_sampled=neg_size, num_classes=vocab_size)
        return tf.reduce_mean(total_loss)


def embedding_layer(x, embedding_shape):
    with tf.variable_scope("embedding"):
        embedding_init = tf.random_uniform(embedding_shape, -1.0, 1.0)
        embedding_matrix = tf.get_variable("E", initializer=embedding_init)
        return tf.nn.embedding_lookup(embedding_matrix, x), embedding_matrix


inputs_emb, inputs_emb_W = embedding_layer(inputs, [vocab_size, embedding_size])

cost = noise_contrastive_loss(inputs_emb, [vocab_size, embedding_size], [vocab_size], labels)

#cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

session = initialize_session()

init = tf.global_variables_initializer()
session.run(init)

saver = tf.train.Saver()

data_point = 0
avg_loss, it_log, it_save, it_sample = .0, 100, 5000, 1000

for epoch in range(100):
    total_cost = 0
    start = time.time()

    for i in range(total_batch):
        _inputs_labels = inputs_labels[data_point: data_point + batch_size]
        data_point = (data_point + batch_size) % total_batch

        batch_x = [_i for _i, _l in _inputs_labels]
        batch_y = [_l for _i, _l in _inputs_labels]

        _, loss = session.run([optimizer, cost], feed_dict={inputs: batch_x, labels: batch_y})

        total_cost += loss

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_cost))
    print(time.time() - start)

save_path = saver.save(session, 'w2v_model')

'''
embeddings = dict()

for i in line:
    temp_a = np.zeros([1, vocab_size])
    temp_a[0][word_to_id[i]] = 1
    temp_emb = session.run([output], feed_dict = {inputs:temp_a})
    temp_emb = np.array(temp_emb)
    embeddings[i] = temp_emb.reshape([vocab_size])
    #print(inputs_emb_W.eval(session=session)[word_to_id[i]])
    embeddings[i] = inputs_emb_W.eval(session=session)[word_to_id[i]]


#save_path = saver.save(session, 'word_embedding_model.ckpt')
json = json.dumps(embeddings)
f=open("embedding.json", "w")
f.write(json)
f.close()

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('
'''