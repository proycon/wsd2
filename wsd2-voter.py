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
    print >> sys.stderr,"Usage: wsd2-voter.py -c [classifierdir1 classifierdir2] -L [lang]  -o [outputdir] -O [timbloptions] -I [divergencefrombestoutputthreshold]"
    

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
divergencefrombestoutputthreshold = 0.9

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
    elif o == '-I':
        divergencefrombestoutputthreshold = float(a)
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
votertestdata = {}
for lemma,pos in testset.lemmas():       
    print >>sys.stderr, "Processing " + lemma.encode('utf-8')
    votertraindata[(lemma,pos)] = {}
    votertestdata[(lemma,pos)] = {}
    
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
            raise Exception("No votertrain found for " + lemma.encode('utf-8') + " in " + classifierdir)              
            

        if os.path.exists(classifierdir + '/' + lemma + '.' + pos + '.votertest'):
            f = codecs.open(classifierdir + '/' + lemma + '.' + pos + '.votertest')
            for line in f:
                line = line.strip()
                fields = line.split('\t')
                if not id in votertestdata[(lemma,pos)]: 
                    votertestdata[(lemma,pos)][id] = {}
                votertestdata[(lemma,pos)][id][classifiername] =  (fields[0], fields[1], fields[2])  #(id, focusword, sense)
            f.close() 
        else:
            raise Exception("No votertest found for " + lemma.encode('utf-8') + " in " + classifierdir)                          
            
#TODO: integrity check?    
    
classifiers = {}
for lemma,pos in testset.lemmas():       
    print >>sys.stderr, "Processing " + lemma.encode('utf-8')
    
    classifiers[(lemma,pos)] = timbl.TimblClassifier(outputdir + '/' + lemma + '.' + pos + '.' + targetlang, timbloptions)
        
    for classifierdir in classifierdirs:  
        classifiername = os.path.basename(classifierdir)
        features = []
        for feature, classlabel in votertraindata[(lemma,pos)][classifiername]:
            features.append(feature)    
        classifiers[(lemma,pos)].append(features, classlabel)
        

print >>sys.stderr, "Training " + str(len(classifiers)) + " classifiers"
for classifier in classifiers:
    classifiers[classifier].train()
    #classifiers[classifier].save()
    
print >>sys.stderr, "Voter Parameter optimisation"
for f in glob.glob(outputdir + '/*.train'):
    os.system("paramsearch ib1 " + f + " > " + f + ".paramsearch")

print >>sys.stderr, "Testing classifiers"
basictimbloptions = timbloptions
for lemma,pos in testset.lemmas():            
    print >>sys.stderr, "Processing " + lemma.encode('utf-8')

    timbloptions = basictimbloptions 
    if os.path.exists(outputdir + '/' + lemma +'.' + pos + '.' + targetlang + '.train.paramsearch'):
        o = wsd2.paramsearch2timblargs(outputdir + '/' + lemma +'.' + pos + '.' + targetlang + '.train.paramsearch')
        timbloptions += " " + o 
        print >>sys.stderr, "Parameter optimisation loaded: " + o
    else:            
        print >>sys.stderr, "NOTICE: No parameter optimisation found!"
     
    out_best = codecs.open(outputdir + '/' + lemma + '.' + pos + '.best','w','utf-8')
    out_oof = codecs.open(outputdir + '/' + lemma + '.' + pos + '.oof','w','utf-8')     
     
    for id in sorted(votertestdata[(lemma,pos)]):     
        features = []
        for classifierdir in classifierdirs:  
            classifiername = os.path.basename(classifierdir)
            features.append( votertestdata[(lemma,pos)][id][classifiername] )
        sense, distribution, distance = classifiers[(lemma,pos)].classify(features)
        print >>sys.stderr, "--> Classifying " + id + " :" + repr(features)
        wsd2.processresult(out_best, out_oof, id, lemma, pos, targetlang, sense, distribution, distance, divergencefrombestoutputthreshold)

    out_best.close()
    out_oof.close()
         
    #score
    os.system('perl ' + wsd2.WSDDIR + '/ScorerTask3.pl ' + outputdir + '/' + lemma + '.' + pos + '.best' + ' ' + wsd2.WSDDIR + '/data/trial/' + targetlang + '/' + lemma + '_gold.txt 2> ' + outputdir + '/' + lemma + '.' + pos + '.best.scorerr')
    os.system('perl ' + wsd2.WSDDIR + '/ScorerTask3.pl ' + outputdir + '/' + lemma + '.' + pos + '.oof' + ' ' + wsd2.WSDDIR + '/data/trial/' + targetlang + '/' + lemma + '_gold.txt -t oof 2> ' + outputdir + '/' + lemma + '.' + pos + '.oof.scorerr')
    
wsd2.scorereport(outputdir)        
