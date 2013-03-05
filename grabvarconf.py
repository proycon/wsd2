#! /usr/bin/env python
# -*- coding: utf8 -*-
import codecs
import os
import sys

variableconfiguration = {}


f = codecs.open(sys.argv[1],'r','utf-8')
for line in f:
    fields = line.strip().split("\t")
    lemma = fields[0]
    id = fields[1]
    var_c = int(id[1])
    var_pos = ('p' in id)
    var_bag = ('b' in id)
    var_lemma = ('l' in id)
    variableconfiguration[lemma] = (id, var_c,var_pos,var_lemma, var_bag)                                
f.close()

for lemma, data in variableconfiguration.items():
    id = data[0]
    os.system("cp ../" + id + "/" + lemma + "* .")    
