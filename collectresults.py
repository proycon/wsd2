#! /usr/bin/env python
# -*- coding: utf8 -*-

import glob
import sys
import os.path
import codecs

basedir = sys.argv[1]

languages = ['nl','es','it','de','fr']

bestdata = {}
oofdata = {}

for lang in languages:
    bestdata[lang] = {}
    oofdata[lang] = {}
    
    for confdir in glob.glob(basedir + '/' + lang + '/c*'):
        if os.path.isdir(confdir):
            conf = os.path.basename(confdir)
            if os.path.exists(confdir + '/results'):
                mode = ''
                f = codecs.open(confdir + '/results','r','utf-8')
                for line in f:
                    line = line.strip()
                    if line:
                        if line[:4] == 'BEST':
                            mode = 'best'
                        elif line[:3] == 'OUT':
                            mode = 'oof'
                        elif line[0] != '-':
                            fields = line.split("\t")
                            lemma = fields[0].strip(':')
                            if lemma == 'AVERAGE':
                                lemma = 'OVERALL'
                            score = float(fields[1])
                            if mode == 'best':
                                if not lemma in bestdata: 
                                    bestdata[lang][lemma] = {}
                                bestdata[lang][lemma][conf] = score
                            elif mode == 'oof':
                                if not lemma in oofdata: 
                                    oofdata[lang][lemma] = {}
                                oofdata[lang][lemma][conf] = score                                
                    f.close()

