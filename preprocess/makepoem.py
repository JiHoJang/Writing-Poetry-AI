import json
import re
from langdetect import detect

def cleanText(readData):
    string = readData.replace('\"', '').replace('_', ' ').replace('&', '').lower()
    string = re.sub(r"[^A-Za-z0-9(),?\'\`_]", " ", string)
    string = re.sub(r"\'s", " &s", string)
    string = re.sub(r"\'ve", " &ve", string)
    string = re.sub(r"n\'t", " n&t", string)
    string = re.sub(r"\'re", " &re", string)
    string = re.sub(r"\'d", " &d", string)
    string = re.sub(r"\'ll", " &ll", string)
    string = re.sub(r"\'m ", " &m ", string)
    string = re.sub(r"\'", "", string)
    string.replace("'", "")
    string = re.sub(r"&", "'", string)
    string = re.sub(r",", "", string)
    #string = re.sub(r"\\!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", "", string)
    #string.replace('!', " ! ").replace('?', " ? ")
    return string

data = []

p = []

with open("unim_poem.json") as json_file:
    data = json.load(json_file)
    for i in range(len(data)):
        try:
            sentences = []
            poem = dict()
            if detect(data[i]['poem']) == 'en':
                sen = data[i]['poem'].split('\n')
                for s in sen:
                    s = cleanText(s)
                    s = s.split()
                    sentences.append(s)
            poem["poem"] = sentences
            p.append(poem)
        except Exception:
            pass


d = []

for i in p:
    # 0 is encode, 1 is prev, 2 is post
    for j in range(1, len(i['poem'])):
        temp = dict()
        temp['encode'] = i['poem'][j]
        temp['decode_pre'] = i['poem'][j-1]
        try:
            temp['decode_pos'] = i['poem'][j+1]
        except Exception:
            print('except')
            temp['decode_pos'] = ["<eos>"]
        d.append(temp)

json2 = json.dumps(d)
f = open("skip_data.json", "w")
f.write(json2)
f.close()