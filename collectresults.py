#! /usr/bin/env python
# -*- coding: utf8 -*-

import glob
import sys
import os.path
import codecs

basedir = sys.argv[1]

languages = ['nl','es','it','de','fr']
confs = set()

bestdata = {}
oofdata = {}

for lang in languages:
    bestdata[lang] = {}
    oofdata[lang] = {}
    
    for confdir in glob.glob(basedir + '/' + lang + '/c*'):
        if os.path.isdir(confdir):
            conf = os.path.basename(confdir)
            confs.add(conf)
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


print "TYPE\tLANG\tLEMMA\tFIRST\tSECOND\tTHIRD",
for conf in sorted(confs):
    print "\t" + conf,
print

for lang in languages:
    if lang in bestdata:
        lemmas = set(bestdata[lang]) | set(oofdata[lang])
        
        for lemma in sorted(lemmas):
            print "best\t" + lang + "\t" + lemma.encode('utf-8')
            if lemma in bestdata[lang]:
                for x in sorted(bestdata[lang][lemma].items(), key=x[1] * -1)[:3]: 
                    print "\t" + x,
                for conf in sorted(confs):
                    if conf in bestdata[lang][lemma]:
                        print "\t" + str(bestdata[lang][lemma][conf]),
                    else:
                        print "\t0",
            else:
                print "-\t-\t-",
                for conf in sorted(confs):
                    print "\t0",
            print
            
            print "oof\t" + lang + "\t" + lemma.encode('utf-8')
            if lemma in oofdata[lang]:
                for x in sorted(oofdata[lang][lemma].items(), key=x[1] * -1)[:3]: 
                    print "\t" + x,
                for conf in sorted(confs):
                    if conf in oofdata[lang][lemma]:
                        print "\t" + str(oofdata[lang][lemma][conf]),
                    else:
                        print "\t0",
            else:
                print "-\t-\t-",
                for conf in sorted(confs):
                    print "\t0",
            print            
            
            

