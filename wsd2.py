#! /usr/bin/env python
# -*- coding: utf8 -*-

import sys
import getopt
import os
import codecs
from lxml import etree as ElementTree
from pynlpl.formats.moses import PhraseTable
import corenlp
import timbl

def usage():
    """Print usage instructions"""
    print >> sys.stderr,"Usage: wsd2.py --train -s [source-text] -t [target-text] -m [moses-phrasetable] -w [targetwords-file]"
    print >> sys.stderr," -c [int]   context size"
    print >> sys.stderr," -l         add lemmatisation features"
    print >> sys.stderr," -p         add PoS-tag features"    
    print >> sys.stderr," -b         Generate Bag-of-Words for keywords global context"
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
            
            
class TestSet(object):
    languages = {
        'english': 'en',
        'french': 'fr',
        'italian': 'it',
        'german': 'de',
        'dutch': 'nl',
    }

    def __init__(self, filenames = []):
        if (isinstance( filenames,str) or isinstance(filenames,unicode)):
            self.load(filenames) 
        else:
            for filename in filenames:
                self.load(filename)
                

    def load(self, filename):  
        """Read test or trial data and parses it into a usable datastructure"""
        tree = ElementTree.parse(filename)
        root = tree.xpath("/corpus")
        if len(root) > 0: 
            root = root[0]
        else:
            raise Exception("This is not a valid test-file!")
        self.lang = TestSet.languages[root.attrib['lang']]
        self.lexunits = {}
        self.orderedlemmas = [] #we have to retain the order somehow, dictionary is unordered

        for lemmanode in root.findall('.//lexelt'):
            lemma, pos = lemmanode.attrib['item'].rsplit(".",1)

            instances = {}
            for instancenode in lemmanode.findall('.//instance'):            
                id = int(instancenode.attrib['id'])
                contextnode = instancenode.find('.//context')
                leftcontext = contextnode.text
                if not leftcontext: leftcontext = ""    
                if not isinstance(leftcontext, unicode):
                    leftcontext = unicode(leftcontext, 'utf-8')
                headnode = contextnode.find('.//head')
                head = headnode.text
                if not isinstance(head, unicode):
                    head = unicode(head, 'utf-8')
                rightcontext = headnode.tail
                if not rightcontext: rightcontext = ""
                if rightcontext and not isinstance(rightcontext, unicode):
                    rightcontext = unicode(rightcontext, 'utf-8')
                instances[id] = (id, leftcontext,head,rightcontext) #room for output classes is reserved (set to None)

            self.lexunits[lemma+'.'+pos] = instances
            self.orderedlemmas.append( (lemma,pos) ) #(so we can keep right ordering)

    def lemmas(self):
        for lemma,pos in self.orderedlemmas:
            yield lemma, pos

    def has(self, lemma, pos):
        return (lemma+'.'+pos in self.lexunits)

    def instances(self, lemma,pos):
        if lemma+'.'+pos in self.lexunits:
            return sorted(self.lexunits[lemma+'.'+pos].items()) #return instances ordered
        else:
            raise KeyError

def loadtargetwords(targetwordsfile):            
    targetwords = set()
    f = codecs.open(targetwordsfile, 'r','utf-8')
    for line in f:
        if line.strip() and line[0] != '#':
            lemma,pos = line.strip().split('\t')
            targetwords.add( (lemma,pos) )
    return targetwords

    
class CLWSD2Trainer(object):    
    
    def __init__(self, outputdir, targetlang, phrasetablefile, sourcefile, targetfile, targetwordsfile, sourcetagger, contextsize, DOPOS, DOLEMMAS, exemplarweights):      
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
        
        
        print >>sys.stderr, "Loading Target Words " + targetwordsfile
        self.targetwords = loadtargetwords(targetwordsfile)
        
     
        self.targetlang = targetlang                        
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
                                    self.classifiers[(sourcelemma,sourcepos, self.targetlang)] = timbl.TimblClassifier(sourcelemma +'.' + sourcepos + '.' + targetlang, self.timbloptions)
                            
                                self.classifiers[(sourcelemma,sourcepos, self.targetlang)].append(features, target)

                     
                    print >>sys.stderr                           

        f_source.close()
        f_target.close()
        
        
        print >>sys.stderr, "Training classifiers"
        for classifier in self.classifiers:
            self.classifiers[classifier].train()
            
        
        #TODO: paramsearch    
        
  
    
class CLWSD2Tester(object):          
    def __init__(self, testdir, outputdir, targetlang,targetwordsfile, sourcetagger, contextsize, DOPOS, DOLEMMAS, timbloptions):        
        self.sourcetagger = sourcetagger
        
        
        print >>sys.stderr, "Loading Target Words " + targetwordsfile       
        self.targetwords = loadtargetwords(targetwordsfile)
        self.testdata = {}
        self.classifiers = {}
        
        self.targetlang = targetlang
        
        self.testdata[(lemma,pos)] = TestSet(testdir+"/" + lemma + '.data')

        self.contextsize = contextsize
        self.DOPOS = DOPOS
        self.DOLEMMAS = DOLEMMAS
        self.exemplarweights = exemplarweights
        self.outputdir = outputdir

        testfiles = []
        for lemma, pos in self.targetwords:
            if os.path.exists(testdir+"/" + lemma + '.data'):
                testfiles.append(testdir+"/" + lemma + '.data')
            else:
                print >>sys.stderr, "WARNING: No testfile found for ", lemma
                
        self.testset = TestSet(testfiles)
        self.timbloptions = timbloptions
              
       
    def run(self):
        print >>sys.stderr, "Extracting features from testset"
        for lemma,pos in self.testdata.lemmas():            
            classifier = timbl.TimblClassifier(lemma +'.' + pos + '.' + self.targetlang, timbloptions)
            out_best = codecs.open(outputdir + '/' + lemma + '.' + pos + '.best','w','utf-8')
            out_oot = codecs.open(outputdir + '/' + lemma + '.' + pos + '.oot','w','utf-8')
                
            for instancenum, (id, leftcontext,head,rightcontext) in enumerate(self.testdata.instances(lemma,pos)):
                print >>sys.stderr, lemma.encode('utf-8') + '.' + pos + " @" + str(instancenum+1)
                
                sourcewords_untok = leftcontext + [head] + rightcontext
                
                sourcewords, sourcepostags, sourcelemmas = sourcetagger.process(sourcewords_untok)
                
                #find new head position (may have moved due to tokenisation)
                origindex = len(leftcontext.split(' '))
                mindistance = 9999
                focusindex = -1
                
                for i, word in enumerate(sourcewords): 
                    if word == head:
                        distance = abs(origindex - i)
                        if distance <= mindistance:
                            focusindex = i
                            mindistance = distance 
                           
                if focusindex == -1: 
                    raise Exception("Focus word not found after tokenisation! This should not happen!")
                         
                #grab local context features
                features = []                    
                for j in range(focusindex - self.contextsize, focusindex + len(sourcewords) + self.contextsize):
                    if j > 0 and j < focusindex + len(sourcewords):
                        features.append(sourceword[j])
                        if self.DOPOS: features.append(sourcepostags[j])
                        if self.DOLEMMAS: features.append(sourcelemmas[j])
                    else:
                        features.append("{NULL}")
                        if self.DOPOS: features.append("{NULL}")
                        if self.DOLEMMAS: features.append("{NULL}")     
                                            
                _, distribution, distance = classifier.classify(features)
                
                bestscore = max(distribution.values())
                bestsenses = [ sense for sense, score in distribution.items() if score == bestscore ]
                tenbestsenses = [ sense for sense, score in sorted(distribution.items()[:10], key=lambda x: -1 * x[1]) ]                                  
                out_best.write(lemma + "." + pos + "." + self.targetlang + ' ' + id + ' :: ' + ';'.join(bestsenses) + ';\n')
                out_oot.write(lemma + "." + pos + "." + self.targetlang + ' ' + id + ' ::: ' + ';'.join(tenbestsenses) + ';\n')
                
                print >>sys.stderr, "\t" + distribution
                
            out_best.close()
            out_oot.close()
                
                 
                
                
        
        
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
            
    
    
    
    
    
