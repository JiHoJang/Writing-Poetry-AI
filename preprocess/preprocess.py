import json
import ndjson
import re
from langdetect import detect

words = []

with open('unim_poem.json') as json_file:
    data = json.load(json_file)
    for p in data:
        if detect(p['poem']) == 'en':
            string = p['poem']
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
            string = re.sub("__LaTex__", "", string)
            specialwords = ['EOS', 'GOO', '__LaTex__']
            toLower = lambda x: " ".join( a if a in specialwords else a.lower() \
                for a in x.split() )
#            print(toLower(string.strip()))
            words += toLower(string.strip()).split()
            
with open('gutenberg-poetry-v001.ndjson') as f:
    data = ndjson.load(f)
    for p in data:
        string = p['s']
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
        string = re.sub("__LaTex__", "", string) # consider how to deal with latex later.
        # prevent some special words from converting to lowercase.
        specialwords = ['EOS', 'GOO', '__LaTex__']
        toLower = lambda x: " ".join( a if a in specialwords else a.lower() \
            for a in x.split() )
        words += toLower(string.strip()).split()

words = list(set(words))

print(len(words))



f = open("output.txt", 'w')
for i in words:
    f.write(i + ' ')
f.close()
