#! /usr/bin/env python
# -*- coding: utf8 -*-

import sys
import os
import codecs

def processfile(filename, targetwords):
    f = codecs.open(filename,'r','utf-8')
    for line in f:
         first, second = line.split(' :: ')
         lemma,pos,lang = first.split(' ').split('.')
         if not (lemma,pos) in targetwords:
            targetwords[(lemma,pos)] = {}
         if not lang in targetwords[(lemma,pos)]:
            targetwords[(lemma,pos)][lang] = set()
         
         for sensedata in second.split(';'):
            sensedata = sensedata.strip()
            if sensedata:
                raw = sensedata.split(' ')
                assert raw[-1].isdigit()
                sense = ' '.join(raw[:-1]).strip()                
                targetwords[(lemma,pos)][lang].add(sense)
    f.close()

targetwords = {}
for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:
        if filename[0] != '.':
            if filename[-5:] == '.gold' or filename[-8:] == 'gold.txt':
                processfile(root + '/' + filename,targetwords)
            
for lemma,pos in targetwords:
    for lang in targetwords[(lemma,pos)]:
        print lemma.encode('utf-8') + '\t' + pos + '\t' + lang + '\t' + ';'.join([ sense.encode('utf-8') for sense in sorted(targetwords[(lemma,pos)][lang])])         
