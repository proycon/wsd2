#! /usr/bin/env python
# -*- coding: utf8 -*-

import getopt
import sys
import os
import timbl
import wsd2
import codecs
import glob


def usage():
    print >> sys.stderr,"Usage: wsd2-voter.py -c [classifierdir1 classifierdir2] -L [lang]  -o [outputdir] -O [timbloptions]"
    

try:
    opts, args = getopt.getopt(sys.argv[1:], "c:L:o:O:")
except getopt.GetoptError, err:
    # print help information and exit:
    print str(err)
    usage()
    sys.exit(2)           


targetlang = None
outputdir = "."
timbloptions = "-a 0 -k 1"
testdir = wsd2.WSDDIR + "/data/trial"
targetwordsfile = wsd2.WSDDIR + "/data/targetwords.trial"
classifierdirs = []

for o, a in opts:
    if o == "-c":	
        classifierdirs = a.split(' ')
    elif o == '-L':
        targetlang = a
    elif o == '-o':
        outputdir = a
    elif o == '-O':
        timbloptions = a
    elif o == '-T':
        testdir = a     
    elif o == '-w':   
        targetwordsfile = a
        
    else:
        raise Exception("Unknown option: " + o)





classifiers = {}
targetwords = wsd2.loadtargetwords(targetwordsfile)
testfiles = []
for lemma, pos in targetwords:
    if os.path.exists(testdir+"/" + lemma + '.data'):
        testfiles.append(testdir+"/" + lemma + '.data')
    else:
        print >>sys.stderr, "WARNING: No testfile found for " + lemma + " (tried " + testdir+"/" + lemma + '.data)'
testset = wsd2.TestSet(testfiles)

votertraindata = {}
for lemma,pos in testset.lemmas():       
    print >>sys.stderr, "Processing " + lemma.encode('utf-8')
    votertraindata[(lemma,pos)] = {}
    
    for classifierdir in classifierdirs:  
        classifiername = os.path.basename(classifierdir)
        
        votertraindata[(lemma,pos)][classifiername] = []
        
        if os.path.exists(classifierdir + '/' + lemma + '.' + pos + '.votertrain'):
            f = codecs.open(classifierdir + '/' + lemma + '.' + pos + '.votertrain')
            for line in f:
                line = line.strip()
                fields = line.split('\t')
                votertraindata[(lemma,pos)][classifiername].append( (fields[0], fields[1]) )  #(train,gold)
            f.close() 
        else:
            print >>sys.stderr, "No votertrain found for " + lemma.encode('utf-8')              
            
#TODO: integrity check?    
    
classifiers = {}
for lemma,pos in testset.lemmas():       
    print >>sys.stderr, "Processing " + lemma.encode('utf-8')
    
    classifiers[(lemma,pos)] = timbl.TimblClassifier(outputdir + '/' + lemma + '.' + pos + '.' + targetlang, timbloptions)
        
    for classifierdir in classifierdirs:  
        classifiername = os.path.basename(classifierdir)
        features = []
        for feature, classlabel in votertraindata[(lemma,pos)]:
            features.append(feature)    
        classifiers[(lemma,pos)].append(features, classlabel)
        

print >>sys.stderr, "Training " + str(len(classifiers)) + " classifiers"
for classifier in classifiers:
    classifiers[classifier].train()
    classifiers[classifier].save()
    
print >>sys.stderr, "Parameter optimisation"
for f in glob.glob(outputdir + '/*.train'):
    os.system("paramsearch ib1 " + f + " > " + f + ".paramsearch")




    
