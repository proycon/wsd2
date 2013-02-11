#! /usr/bin/env python
# -*- coding: utf8 -*-

import sys
import getopt
import os
import codecs
from pynlpl.formats.moses import PhraseTable
import corenlp
import timbl

def usage():
    """Print usage instructions"""
    print >> sys.stderr,"Usage: wsd2.py --train -s [source-text] -t [target-text] -m [moses-phrasetable] -w [targetwords-file]"
    print >> sys.stderr," -c [int]   context size"
    print >> sys.stderr," -l         add lemmatisation features"
    print >> sys.stderr," -p         PoS-tag"    
    print >> sys.stderr," -b         Generate Bag-of-Words"
    print >> sys.stderr," -B [absolute_threshold,probability_threshold,filter_threshold]" #TODO: implement
    print >> sys.stderr,"                        Parameters for the selection of bag of words."            
    print >> sys.stderr,"          1) Bag-of-word needs to occur at least x times in context"
    print >> sys.stderr,"          2) Bag-of-word needs to have sense|keyword probability of at least x"
    print >> sys.stderr,"          3) Filter out words with a global corpus occurence less than x"
    print >> sys.stderr," -R                     Automatically compute absolute threshold per word-expert"
    print >> sys.stderr," --Stagger   Tagger for source language, set to frog:[port] or freeling:[channel] or corenlp, start the tagger server manually first for the first two"
    #print >> sys.stderr," --Ttagger   Tagger for target language, set to frog:[port] or freeling:[channel], start the tagger server manually first"
    
class Tagger(object):    
     def __init__(self, *args):        
        self.tagger = None
        if args[0] == "frog":
            self.mode = "frog"
            self.port = int(args[1])
        elif args[0] == "freeling":
            self.mode = "freeling"
            self.channel = int(args[1])
        elif args[0] == "corenlp":
            self.mode = "corenlp"
            self.tagger = corenlp.StanfordCoreNLP()            
        else:
            raise Exception("Invalid mode: " + args[0])
        
     def process(self, words):
        if self.mode == "frog":
            #TODO
            raise NotImplemented
        elif self.mode == "freeling":
            #TODO
            raise NotImplemented
        elif self.mode == "corenlp":            
            data = self.tagger.parse(" ".join(words))
            words = []
            postags = []
            lemmas = []
            for sentence in data['sentences']:
                for word, worddata in sentence['words']:
                    words.append(word)
                    lemmas.append(worddata['Lemma'])
                    postags.append(worddata['PartOfSpeech'])
            return words, pos, lemmas
            
            

            

    
class CLWSD2Trainer(object):    
    
    def __init__(self, outputdir, phrasetablefile, sourcefile, targetfile, targetwordsfile, sourcetagger, targettagger, contextsize, DOPOS, DOLEMMAS, exemplarweights):      
        if not os.path.exists(phrasetablefile):
            raise Exception("Moses phrasetable does not exist: " + phrasetablefile)
        if not os.path.exists(sourcefile):
            raise Exception("Sourcefile does not exist: " + sourcefile)
        if not os.path.exists(targetfile):
            raise Exception("Target file does not exist: " + targetfile)
        if not os.path.exists(targetwordsfile):
            raise Exception("Target words file does not exist: " + targetwordsfile)                             
        self.sourcefile = sourcefile
        self.targetfile = targetfile
        
        self.sourcetagger = sourcetagger
        self.targettagger = targettagger
        
        print >>sys.stderr, "Loading Target Words " + targetwordsfile
        self.targetwords = set()
        f = codecs.open(targetwordsfile, 'r','utf-8')
        for line in f:
            if line.strip() and line[0] != '#':
                lemma,pos = line.strip().split('\t')
                self.targetwords.add( (lemma,pos) )             
                        
        print >>sys.stderr, "Loading Moses Phrasetable " + phrasetablefile
        self.phrasetable = PhraseTable(phrasetablefile)
        
        self.contextsize = contextsize
        self.DOPOS = DOPOS
        self.DOLEMMAS = DOLEMMAS
        self.exemplarweights = exemplarweights
        self.outputdir = outputdir
        self.classifiers = {}
        
    def run():        
        print >>sys.stderr, "Reading texts and extracting features"
        f_source = codecs.open(self.sourcefile,'r','utf-8')
        f_target = codecs.open(self.targetfile,'r','utf-8')
        
        for sentencenum, sourceline, targetline in enumerate(zip(f_source, f_target)):
            sourceline = sourceline.strip()
            targetline = targetline.strip()
            sourcewords = sourceline.split()
            targetwords = targetline.split()
            
            sourcewords, sourcepostags, sourcelemmas = sourcetagger.process(sourcewords)
            #targetpostags, targetlemmas = targettagger.process(targetwords)            
            
            for i, (sourceword, sourcepos, sourcelemma) in enumerate(zip(sourcewords, sourcepos, sourcelemma)):                

                if (sourcelemma, sourcepos) in targetwords and sourceword in phrasetable:
                    
                    print >>sys.stderr, "@" + str(sentencenum+1) + ":" + str(i) + " -- Found " + sourcelemma.encode('utf-8') + '.' + sourcepos,
                    #grab local context features
                    features = []                    
                    for j in range(i - self.contextsize, i + len(sourcewords) + self.contextsize):
                        if j > 0 and j < i + len(sourcewords):
                            features.append(sourceword[j])
                            if self.DOPOS: features.append(sourcepostags[j])
                            if self.DOLEMMAS: features.append(sourcelemmas[j])
                        else:
                            features.append("{NULL}")
                            if self.DOPOS: features.append("{NULL}")
                            if self.DOLEMMAS: features.append("{NULL}")                            
                    
                    
                    #find options in phrasetable
                    translationoptions = phrasetable[sourceword]  #[ (target, Pst, Pts, null_alignments) ]
                    
                    #which of the translation options actually occurs in the target sentence?
                    for target, Pst, Pts,_ in translationoptions:
                        
                        targetword = ""
                        
                        found = False
                        n = len(target.split(" "))
                        for j in range(0,len(targetwords)):
                            if " ".join(targetwords[j:j+n]) == target:
                                found = True
                                print >>sys.stderr, "\t" + targetword.encode('utf-8')                                
                                if not (sourcelemma,sourcepos) in self.classifiers:
                                    self.classifiers[(sourcelemma,sourcepos)] = timbl.TimblClassifier(sourcelemma +'.' + sourcepos, self.timbloptions)
                            
                                self.classifiers[(sourcelemma,sourcepos)].append(features, target)

                     
                    print >>sys.stderr                           

        f_source.close()
        f_target.close()
                    
                    
    
    
if __name__ == "__main__":
    try:
	    opts, args = getopt.getopt(sys.argv[1:], "s:t:c:lpbB:R", ["train","Stagger=","Ttagger="])
    except getopt.GetoptError, err:
	    # print help information and exit:
	    print str(err)
	    usage()
	    sys.exit(2)           
    
    TRAIN = TEST = False
    sourcefile = targetfile = phrasetablefile = ""
    DOLEMMAS = False
    DOPOS = False
    sourcetagger = None
    targettagger = None
    	    
    for o, a in opts:
        if o == "--train":	
            TRAIN = True
        elif o == "-s":
            sourcefile = a
            if not os.path.exists(sourcefile):
                print >>sys.stderr, "ERROR: Source file " + sourcefile + " does not exist"
                sys.exit(2)
        elif o == "-t":
            targetfile = a
            if not os.path.exists(targetfile):
                print >>sys.stderr, "ERROR: Targetfile " + targetfile + " does not exist"
                sys.exit(2)            
        elif o == "-m":
            phrasetablefile = a
        elif o == '-p':
            DOPOS = True
        elif o == '-l':            
            DOLEMMAS = True
        elif o == "--Stagger":
            sourcetagger = Tagger(*a.split(':'))
        elif o == "--Ttagger":
            targettagger = Tagger(*a.split(':'))
        else: 
            print >>sys.stderr,"Unknown option: ", o
            sys.exit(2)
            
    
    
    
    
    
