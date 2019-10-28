import json
import ndjson
import re
from langdetect import detect

words = []

num_poem = 0

with open('unim_poem.json') as json_file:
    data = json.load(json_file)
    for p in data:
#        print(p['poem'].split('\n')[0].strip()))
        temp = p['poem'].replace('_', ' ')
        try:
            if detect(temp) == 'en':
                string = temp.replace('\"', '').replace('_', ' ').replace(',', ' , ').replace('!', ' ! ').replace(':', ' : ').lower()
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

                words += string.split()
                num_poem += 1
            else :
                print('not english')
        except Exception:
            pass

print('Complete unim_poem')  
with open('gutenberg-poetry-v001.ndjson') as f:
    data = ndjson.load(f)
    for p in data:
        string = p['s'].replace('\"', '').replace('_', ' ').replace(',', ' , ').replace('!', ' ! ').replace(':', ' : ').lower()
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

        words += string.split()
words = list(set(words))

print(len(words))



f = open("output2.txt", 'w')
for i in words:
    f.write(i + ' ')
f.close()
