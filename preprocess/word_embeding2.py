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
    string = readData.replace("\n", "<eos>\n").replace('\"', '').replace('_', ' ').lower()
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

def initialize_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

sentences = []

'''
with open("../preprocess/multim_poem.json") as json_file:
#with open("multim_poem.json") as json_file:
    data = json.load(json_file)
    for i in data:
        sentences.append(cleanText(i['poem']).split())
'''

with open("unim_poem.json") as json_file:
#with open("unim_poem.json") as json_file:
    data = json.load(json_file)
    for i in range(len(data)):
        try:
            if detect(data[i]['poem']) == 'en':
                temp = cleanText(data[i]['poem']).split('\n')
                for s in temp:
                    sentences.append(s.split())
                    words += s.split()
        except Exception:
            pass

'''
with open("unim_poem.json") as json_file:
#with open("unim_poem.json") as json_file:
    data = json.load(json_file)
    for i in range(len(data)):

        temp = cleanText(data[i]['poem'])
'''


print("Complete open unim_poem.json")

with open("gutenberg-poetry-v001.ndjson") as f:
    data = ndjson.load(f)
    for i in range(100000):
        temp = cleanText(data[i]['s']).split('\n')
        for s in temp:
            sentences.append(s.split())
            words += s.split()

print("Complete open gutenberg")

words = list(set(words))

print(len(words))

f = open("output2.txt", 'w')
for i in words:
    f.write(i + ' ')
f.close()

f = open("output2.txt", "r")
#f = open("output.txt", "r")
line = f.readline().split()
#line += '\n'

word_to_id, id_to_word = mapping(line)
f.close()

json = json.dumps(word_to_id)
f=open("wri.json", "w")
f.write(json)
f.close()

data = []

WINDOW_SIZE = 4
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
print("Complete make sentences")


batch_size = 100
total_batch = int(vocab_size / batch_size)
embedding_size = 500

tf.reset_default_graph()


inputs = tf.placeholder(tf.int32, shape=[None,])
labels = tf.placeholder(tf.int32, shape=[None,])


inputs_emb_W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
inputs_emb = tf.nn.embedding_lookup(inputs_emb_W, inputs)
W_T = tf.Variable(tf.random_uniform([embedding_size, vocab_size], -1.0, 1.0))
W_b = tf.Variable(tf.zeros([vocab_size]))
output = tf.matmul(inputs_emb, W_T) + W_b
'''

w1 = tf.Variable(tf.random_normal([vocab_size, embedding_size]), dtype = tf.float32)
b1 = tf.Variable(tf.random_normal([embedding_size]), dtype = tf.float32)
w2 = tf.Variable(tf.random_normal([embedding_size, vocab_size], dtype = tf.float32))
b2 = tf.Variable(tf.random_normal([vocab_size]), dtype = tf.float32)

hidden_y = tf.matmul(input, w1) + b1
output = tf.matmul(hidden_y, w2) + b2
'''

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels))

optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)

session = initialize_session()

init = tf.global_variables_initializer()
session.run(init)

saver = tf.train.Saver()

for epoch in range(1):
    total_cost = 0
    start = time.time()
    for i in range(total_batch):

        batch_x = skip_grams_inputs[i*batch_size:(i+1)*batch_size]
        batch_y = skip_grams_labels[i * batch_size:(i + 1) * batch_size]

        _, loss = session.run([optimizer, cost], feed_dict = {inputs: batch_x, labels:batch_y})
        total_cost += loss

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_cost))
    print(time.time() - start)

#save_path = saver.save(session, 'word_embedding_model')

embeddings = dict()

for i in line:
    '''
    temp_a = np.zeros([1, vocab_size])
    temp_a[0][word_to_id[i]] = 1
    temp_emb = session.run([output], feed_dict = {inputs:temp_a})
    temp_emb = np.array(temp_emb)
    embeddings[i] = temp_emb.reshape([vocab_size])
    '''
    #print(inputs_emb_W.eval(session=session)[word_to_id[i]])
    embeddings[i] = inputs_emb_W.eval(session=session)[word_to_id[i]]


#save_path = saver.save(session, 'word_embedding_model.ckpt')
json = json.dumps(embeddings)
f=open("embedding.json", "w")
f.write(json)
f.close()

'''
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('
'''