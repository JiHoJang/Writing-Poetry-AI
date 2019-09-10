import json

words = []

with open('multim_poem.json') as json_file:
    data = json.load(json_file)
    for p in data:
        words += p['poem'].split()

with open('unim_poem.json') as json_file:
    data = json.load(json_file)
    for p in data:
        words += p['poem'].split()

words = list(set(words))

f = open("output.txt", 'w')
for i in words:
    f.write(i + ' ')
f.close()
