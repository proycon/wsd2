#! /usr/bin/env python
# -*- coding: utf8 -*-

import codecs
import os
data = {}

f = codecs.open('data/targetwords','r','utf-8')
for line in f:
    fields = line.split('\t')
    lang = fields[2]
    if not lang in data:
        data[lang] = set()
    data[lang] = data[lang] | set(fields[3].split(';'))    
f.close()

for lang, senses in data.items():
    if not os.path.exists('data/' + lang + '.variants'):
        f = codecs.open('data/' + lang + '.variants','w')
        for sense in sorted(senses):
            f.write(sense + '\n')
        f.close()    
        
