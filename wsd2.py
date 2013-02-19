#! /usr/bin/env python
# -*- coding: utf8 -*-

import sys
import getopt
import os
import codecs
from lxml import etree as ElementTree
from pynlpl.formats.moses import PhraseTable
from pynlpl.tagger import Tagger
import timbl
import glob

WSDDIR = os.path.dirname(__file__)

def usage():
    """Print usage instructions"""
    print >> sys.stderr,"Usage: wsd2.py --train -L [lang] -s [source-text] -t [target-text] -m [moses-phrasetable] -w [targetwords-file] -o [outputdir] -O [timbloptions]"
    print >> sys.stderr,"       wsd2.py --test -L [lang] -T [testdir] -w [targetwords-file] -o [outputdir] -O [timbloptions]"
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
    print >> sys.stderr," --Ttagger   Tagger for target language, set to frog:[port] or freeling:[channel] (start the tagger server manually first) or  de.lex or fr.lex for built-in lexicons.. "
        
class TestSet(object):
    languages = {
        'english': 'en',
        'french': 'fr',
        'italian': 'it',
        'german': 'de',
        'dutch': 'nl',
    }

    def __init__(self, filenames = []):
        self.lexunits = {}
        self.orderedlemmas = [] #we have to retain the order somehow, dictionary is unordered
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
    targetwords = {}
    f = codecs.open(targetwordsfile, 'r','utf-8')
    for line in f:
        if line.strip() and line[0] != '#':
            fields = line.strip().split('\t')
            if len(fields) == 4:
                lemma,pos,lang,senses = fields   
                lemma = lemma.strip()
                pos = pos.strip()         
                if not (lemma,pos) in targetwords: 
                    targetwords[(lemma,pos)] = {}
                targetwords[(lemma,pos)][lang] = senses.split(';')
            elif len(fields) == 2:
                lemma,pos = fields
                lemma = lemma.strip()
                pos = pos.strip()                    
                targetwords[(lemma,pos)] = True
            else:
                raise Exception("Invalid format in targetwords")
    f.close()
                
    return targetwords

def targetmatch(target, senses): 
    target = target.lower()
    for sense in senses:
        if target == sense.lower():
            return sense
    if not ' ' in target:
        for sense in senses:
            if target[:-len(sense) - 1] == sense.lower()[:-1] and len(sense) - len(target) <= 6:  #very crude stem check 
                return sense
    return None
        

    
class CLWSD2Trainer(object):    
    
    def __init__(self, outputdir, targetlang, phrasetablefile, sourcefile, targetfile, targetwordsfile, sourcetagger, targettagger, contextsize, DOPOS, DOLEMMAS, exemplarweights, timbloptions, bagofwords, compute_bow_params, bow_absolute_threshold, bow_prob_threshold, bow_filter_threshold, maxdivergencefrombest = 0.5):      
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
        self.targetwords = loadtargetwords(targetwordsfile)
        print >>sys.stderr, len(self.targetwords),"loaded"
     
        self.targetlang = targetlang                        
        print >>sys.stderr, "Loading Moses Phrasetable " + phrasetablefile
        self.phrasetable = PhraseTable(phrasetablefile)
        
        self.contextsize = contextsize
        self.DOPOS = DOPOS
        self.DOLEMMAS = DOLEMMAS
        self.exemplarweights = exemplarweights
        self.outputdir = outputdir
        self.classifiers = {}
        
        self.bagofwords = bagofwords
        self.compute_bow_params = compute_bow_params
        self.bow_absolute_threshold = bow_absolute_threshold
        self.bow_prob_threshold = bow_prob_threshold
        self.bow_filter_threshold = bow_filter_threshold
        self.timbloptions = timbloptions
        self.maxdivergencefrombest = maxdivergencefrombest

    def probability_sense_given_keyword(self, focuslemma,focuspos,senselabel, lemma,pos, count, totalcount):
        if not focuslemma+'.'+focuspos in count:
            print "focusword not seen", focuslemma +'.'+ focuspos
            return 0 #focus word has not been counted for

        if not senselabel in count[focuslemma+'.'+focuspos]:
            print "sense not seen:", senselabel
            return 0 #sense has never been seen for this focus word

        if not lemma+'.'+pos in totalcount:
            print "keyword not seen:", lemma+'.'+pos
            return 0 #keyword has never been seen

        Ns_kloc = 0.0
        if lemma+'.'+pos in count[focuslemma+'.'+focuspos][senselabel]:
            Ns_kloc = float(count[focuslemma+'.'+focuspos][senselabel][lemma+'.'+pos])        

        Nkloc = 0
        for sense in count[focuslemma+'.'+focuspos]:
            if lemma+'.'+pos in count[focuslemma+'.'+focuspos][sense]:
                Nkloc += count[focuslemma+'.'+focuspos][sense][lemma+'.'+pos]
                

        Nkcorp = float(totalcount[lemma+'.'+pos]) #/ float(totalcount_sum)

        #if focuslemma == 'wild':
        #    print "p = (",Ns_kloc,"/",Nkloc,") * 1/",Nkcorp, " = ",  (Ns_kloc / Nkloc) * (1/Nkcorp)

        return (Ns_kloc / float(Nkloc)) * (1/Nkcorp)


    def make_bag_of_words(self, focuslemma, focuspos, bow_absolute_threshold, count, totalcount):
        print >>sys.stderr, "Computing and writing bag for ",focuslemma.encode('utf-8') + '.' + focuspos,"..."

        if not focuslemma+'.'+focuspos in count:
            return [] #focus word has not been counted for

        bag = []
        #select all words that occur at least 3 times for a sense, and have a probability_sense_given_keyword >= 0.001
        for sense in count[(focuslemma,focuspos)]:
            for lemma, pos in [ x.rsplit('.',1) for x in count[(focuslemma,focuspos)][sense].keys() ]:
                 if (totalcount[(lemma,pos)] >= self.bow_filter_threshold): #filter very rare words (occuring less than 20 times)
                     if count[(focuslemma,focuspos)][sense][(lemma,pos)] >= bow_absolute_threshold:
                        p = self.probability_sense_given_keyword(focuslemma,focuspos,sense,lemma,pos, count, totalcount) 
                        if p >= self.bow_prob_threshold:
                            bag.append( (lemma,pos, sense, count[(focuslemma,focuspos)][sense][(lemma,pos)], p) )

        bag = sorted(bag)
        f = codecs.open(self.outputdir+ '/' + focuslemma + '.' + focuspos+ '.' + self.targetlang + '.bag','w','utf-8')
        for lemma,pos, sense, c, p in bag:
            f.write(lemma + '\t' + pos + '\t' + sense + '\t' + str(c) + '\t' + str(p) + '\n')
        f.close()

        return bag


        

    def run(self):        

        count = {}
        totalcount = {}
        bags = {} #will store bags of words
                             
        if self.bagofwords:
            finalstage = False
        else:
            finalstage = True
        
        
        

        
        while True:
            if self.sourcetagger: self.sourcetagger.reset()
            if self.targettagger: self.targettagger.reset()
            
            if not finalstage and self.bagofwords:
                print >>sys.stderr, "Reading texts and counting for global context (first pass)"                
            elif finalstage:
                print >>sys.stderr, "Reading texts and extracting features (last pass)"
            f_source = codecs.open(self.sourcefile,'r','utf-8')
            f_target = codecs.open(self.targetfile,'r','utf-8')
            for sentencenum, (sourceline, targetline) in enumerate(zip(f_source, f_target)):                    
                print >>sys.stderr, " @" + str(sentencenum+1)            
                
                sourceline = sourceline.strip()
                targetline = targetline.strip()
                sourcewords = sourceline.split()
                targetwords = targetline.split()
                
                sourcewords, sourcepostags, sourcelemmas = self.sourcetagger.process(sourcewords)
                sourcepostags = [ x[0].lower() for x in sourcepostags ]
                               
                if self.targettagger: 
                    targetwords, targetpostags, targetlemmas = self.targettagger.process(targetwords)
                    targetpostags = [ x[0].lower() for x in targetpostags ]                               
                else:
                    targetpostags = []
                    targetlemmas = []
 
                for i, (sourceword, sourcepos, sourcelemma) in enumerate(zip(sourcewords, sourcepostags, sourcelemmas)):                
                    if (sourcelemma, sourcepos) in self.targetwords and sourceword in self.phrasetable:
                                            
                        #find options in phrasetable
                        try:
                            translationoptions = self.phrasetable[sourceword]  #[ (target, Pst, Pts, null_alignments) ]
                        except KeyError:
                            continue
                        
                        
                        print >>sys.stderr, " @" + str(sentencenum+1) + ":" + str(i) + " -- Found " + sourcelemma.encode('utf-8') + '.' + sourcepos,
                        
                        #grab local context features
                        localfeatures = [] 
                        for j in range(i - self.contextsize, i + 1 + self.contextsize):
                            if j > 0 and j < len(sourcewords):
                                localfeatures.append(sourcewords[j])
                                if self.DOPOS: localfeatures.append(sourcepostags[j])
                                if self.DOLEMMAS: localfeatures.append(sourcelemmas[j])
                            else:
                                localfeatures.append("{NULL}")
                                if self.DOPOS: localfeatures.append("{NULL}")
                                if self.DOLEMMAS: localfeatures.append("{NULL}")                            
                        

                        
                        foundtranslationoptions = []
                        bestscore = 0
                        for target, Pst, Pts,_ in translationoptions:
                            #check if and where it occurs in target sense
                            foundindex = -1
                            if ' ' in target:
                                targetl = target.split(' ')
                                for j in range(0,len(targetwords) - len(targetl)):
                                    if targetwords[j:j+len(targetl)] == targetl: 
                                        foundindex = j
                                        break                            
                            else:
                                for j, w in enumerate(targetwords):
                                    if target == w:
                                        foundindex = j
                                        break
                            
                            if foundindex != -1:
                                foundtranslationoptions.append( (target, Pts, foundindex) )
                                if Pts > bestscore: bestscore = Pts
                        
                        #prune translation options scoring too low
                        foundtranslationoptions = [ x for x in foundtranslationoptions if x[1] >= bestscore * self.maxdivergencefrombest ]
                        
                        
                        #which of the translation options actually occurs in the target sentence?
                        for target, Pts,foundindex in foundtranslationoptions:

                            #check if and where it occurs in target sense
                            foundindex = -1
                            if ' ' in target:
                                targetl = target.split(' ')
                                for j in range(0,len(targetwords) - len(targetl)):
                                    if targetwords[j:j+len(targetl)] == targetl: 
                                        foundindex = j
                                        break                            
                            else:
                                for j, w in enumerate(targetwords):
                                    if target == w:
                                        foundindex = j
                                        break
                                                    
                            if foundindex != -1: 
                                #get lemmatised form of target word
                                if self.targettagger:
                                    if ' ' in target:
                                        target = ' '.join(targetlemmas[foundindex:foundindex+len(targetl)])
                                    else:
                                        target = targetlemmas[foundindex] 
                                
                                print >>sys.stderr, "\t\"" + target.encode('utf-8') + "\"",
                                if finalstage:
                                    if not (sourcelemma,sourcepos, self.targetlang) in self.classifiers:
                                        #init classifier
                                        self.classifiers[(sourcelemma,sourcepos, self.targetlang)] = timbl.TimblClassifier(self.outputdir + '/' + sourcelemma +'.' + sourcepos + '.' + targetlang, self.timbloptions)
                                
                                    
                                    if self.bagofwords and (sourcelemma,sourcepos) in bags:                                        
                                        globalfeatures = []
                                        #create new bag
                                        bag = {}
                                        for keylemma,keypos,_,_,_ in bags[(sourcelemma, sourcepos)]:
                                            bag[keylemma,keypos] = 0
                                        
                                        #now count the words in our context
                                        for j, (contextword, contextpos, contextlemma) in enumerate(zip(sourcewords, sourcepostags, sourcelemmas)):
                                            if (contextlemma, contextpos) in bag:
                                                bag[(contextlemma,contextpos)] = 1

                                        #and output the bag of words features
                                        for contextlemma, contextpos in sorted(bag.keys()):
                                            globalfeatures.append(bag[(contextlemma,contextpos)])
                                        
                                        self.classifiers[(sourcelemma,sourcepos, self.targetlang)].append(localfeatures + globalfeatures, target)
                                        
                                    else:
                                        self.classifiers[(sourcelemma,sourcepos, self.targetlang)].append(localfeatures, target)    
                                        
                                elif self.bagofwords:
                                    if (sourcelemma,sourcepos) in count:
                                        if not target in count[(sourcelemma, sourcepos)]:
                                            count[(sourcelemma,sourcepos)][target] = {}
                                        
                                        for j, (contextword, contextpos, contextlemma) in enumerate(zip(sourcewords, sourcepostags, sourcelemmas)):
                                            if j != i:
                                                if not (contextlemma, contextpos) in count[(sourcelemma,sourcepos)][target]:
                                                    count[(sourcelemma, sourcepos)][target][(contextlemma,contextpos)] = 1
                                                else:
                                                    count[(sourcelemma, sourcepos)][target][(contextlemma,contextpos)] += 1
                         
                        print >>sys.stderr                           

            f_source.close()
            f_target.close()
            if finalstage:
                break
            elif self.bagofwords:
                for lemma,pos in count.keys():        
                    if self.compute_bow_params:
                        bags[(lemma,pos)] = self.make_bag_of_words(lemma,pos, self.bow_absolute_threshold,count, totalcount)            
                        absthreshold = self.bow_absolute_threshold
                        if len(bags[(lemma,pos)]) <= 5:
                            #too few results, loosen parameters
                            while len(bags[(lemma,pos)]) <= 5 and absthreshold > 1:
                                absthreshold = absthreshold - 1
                                bags[(lemma,pos)] = self.make_bag_of_words(lemma,pos,absthreshold, count, totalcount)    
                        elif len(bags[(lemma,pos)]) >= 500:
                            #too many results, tighten parameters
                            while len(bags[(lemma,pos)]) >= 500:
                                absthreshold = absthreshold + 1
                                bags[(lemma,pos)] = self.make_bag_of_words(lemma,pos, absthreshold, count, totalcount)                            
                    else:
                        bags[(lemma,pos)] = self.make_bag_of_words(lemma,pos, self.bow_absolute_threshold,  count, totalcount)                
                finalstage = True
        
        self.run2() 
                
            
    def run2(self):        
        print >>sys.stderr, "Training " + str(len(self.classifiers)) + " classifiers"
        for classifier in self.classifiers:
            self.classifiers[classifier].train()

        print >>sys.stderr, "Parameter optimisation"
        for f in glob.glob(self.outputdir + '/*.train'):
            os.system("paramsearch ib1 " + f + " > " + f + ".paramsearch")
        
        
def paramsearch2timblargs(filename):
    f = open(filename)
    for line in f:
        opts = ""
        for field in line.split("."):
            if field in ("IB1","IG","TRIBL","IB2","TRIBL2"):
                opts += "-a " + field
            elif field in ("M","C","D","DC","L","J","N","I","O"):
                opts += " -m " + field
            elif field in ("nw","gr","ig","x2","sv"):
                opts += " -w " + field
            elif field in ("Z","IL") or field[0:3] == "ED:":
                opts += " -d " + field
            elif len(field) >= 2 and field[0] == "L" and field[1:].isdigit():
                opts += " -L " + field[1:]                
            elif len(field) >= 2 and field[0] == "k" and field[1:].isdigit():
                opts += " -k " + field[1:]        
        print opts
    f.close()
    return opts

    
class CLWSD2Tester(object):          
    def __init__(self, testdir, outputdir, targetlang,targetwordsfile, sourcetagger, timbloptions, contextsize, DOPOS, DOLEMMAS, bagofwords):        
        self.sourcetagger = sourcetagger
        
        
        print >>sys.stderr, "Loading Target Words " + targetwordsfile       
        self.targetwords = loadtargetwords(targetwordsfile)
        self.classifiers = {}
        
        self.targetlang = targetlang
        
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
                print >>sys.stderr, "WARNING: No testfile found for " + lemma + " (tried " + testdir+"/" + lemma + '.data)'
                
        self.testset = TestSet(testfiles)
        self.timbloptions = timbloptions
        self.bagofwords = bagofwords
              
       
    def run(self):
        global WSDDIR
        print >>sys.stderr, "Extracting features from testset"
        for lemma,pos in self.testset.lemmas():            
            print >>sys.stderr, "Processing " + lemma.encode('utf-8')

            timbloptions = self.timbloptions 
            if os.path.exists(self.outputdir + '/' + lemma +'.' + pos + '.' + self.targetlang + '.train.paramsearch'):
                timbloptions += " " + paramsearch2timblargs(self.outputdir + '/' + lemma +'.' + pos + '.' + self.targetlang + '.train.paramsearch')            
            
            classifier = timbl.TimblClassifier(self.outputdir + '/' + lemma +'.' + pos + '.' + self.targetlang, timbloptions)
            out_best = codecs.open(outputdir + '/' + lemma + '.' + pos + '.best','w','utf-8')
            out_oof = codecs.open(outputdir + '/' + lemma + '.' + pos + '.oof','w','utf-8')
                
            for instancenum, (id, leftcontext,head,rightcontext) in enumerate(self.testset.instances(lemma,pos)):
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
                        features.append(sourcewords[j])
                        if self.DOPOS: features.append(sourcepostags[j])
                        if self.DOLEMMAS: features.append(sourcelemmas[j])
                    else:
                        features.append("{NULL}")
                        if self.DOPOS: features.append("{NULL}")
                        if self.DOLEMMAS: features.append("{NULL}")     
                                            
                _, distribution, distance = classifier.classify(features)
                
                bestscore = max(distribution.values())
                bestsenses = [ sense for sense, score in distribution.items() if score == bestscore ]
                fivebestsenses = [ sense for sense, score in sorted(distribution.items()[:5], key=lambda x: -1 * x[1]) ]                                  
                out_best.write(lemma + "." + pos + "." + self.targetlang + ' ' + id + ' :: ' + ';'.join(bestsenses) + ';\n')
                out_oof.write(lemma + "." + pos + "." + self.targetlang + ' ' + id + ' ::: ' + ';'.join(fivebestsenses) + ';\n')
                
                print >>sys.stderr, "\t" + distribution
                
            out_best.close()
            out_oof.close()
                
            #score
            os.system(WSDDIR + '/ScorerTask3.pl ' + outputdir + '/' + lemma + '.' + pos + '.best' + ' data/trial/' + self.targetlang + '/' + lemma + '_gold.txt')
            os.system(WSDDIR + '/ScorerTask3.pl ' + outputdir + '/' + lemma + '.' + pos + '.oof' + ' data/trial/' + self.targetlang + '/' + lemma + '_gold.txt -t oof')
                 
        
        
if __name__ == "__main__":
    try:
	    opts, args = getopt.getopt(sys.argv[1:], "s:t:c:lpbB:Ro:w:L:O:m:T:", ["train","test", "Stagger=","Ttagger="])
    except getopt.GetoptError, err:
	    # print help information and exit:
	    print str(err)
	    usage()
	    sys.exit(2)           
    
    TRAIN = TEST = False
    sourcefile = targetfile = phrasetablefile = ""
    targetwordsfile = WSDDIR + "/data/targetwords.trial"
    DOLEMMAS = False
    DOPOS = False
    sourcetagger = None
    targettagger = None
    outputdir = "."
    testdir = WSDDIR + "/data/trial"
    targetlang = ""
    exemplarweights = False
    timbloptions = "-a 0 -k 1"
    contextsize = 0
    
    bagofwords = False
    compute_bow_params = False
    bow_absolute_threshold = 3 #Bag-of-word needs to occur at least x times in context
    bow_prob_threshold = 0.001 #Bag-of-word needs to have sense|keyword probability of at least x
    bow_filter_threshold = 20 #Filter out words with a global corpus occurence less than x
    
    for o, a in opts:
        if o == "--train":	
            TRAIN = True
        elif o == "--test":	
            TEST = True            
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
        elif o == '-o':
            outputdir = a
        elif o == '-w':
            targetwordsfile = a
        elif o == '-c':
            contextsize = int(a)
        elif o == '-L':
            targetlang = a   
        elif o == '-T':
            testdir = a
        elif o == '-O':
            timbloptions = a
        elif o == '-b':
            bagofwords = True
        elif o == "-B":
            fields = a.split(",")
            if len(fields) >= 1:
               bow_absolute_threshold = int(fields[0])
            if len(fields) >= 2:
               bow_prob_threshold = float(fields[1])
            if len(fields) >= 3:
               bow_filter_threshold = int(fields[2])
        elif o == '-R':             
            compute_bow_params = True
        else: 
            print >>sys.stderr,"Unknown option: ", o
            sys.exit(2)
            
    if not targetlang:            
        print >>sys.stderr, "ERROR: No target language specified"
        sys.exit(2)
    elif not sourcetagger:            
        print >>sys.stderr, "ERROR: No source tagger specified"
        sys.exit(2)            
        
    if TRAIN:
        if not phrasetablefile:
            print >>sys.stderr, "ERROR: No phrasetable file specified"
            sys.exit(2)
        elif not targettagger:            
            print >>sys.stderr, "WARNING: No target tagger specified"
        trainer = CLWSD2Trainer(outputdir, targetlang, phrasetablefile, sourcefile, targetfile, targetwordsfile, sourcetagger, targettagger, contextsize, DOPOS, DOLEMMAS, exemplarweights, timbloptions, bagofwords,compute_bow_params, bow_absolute_threshold, bow_prob_threshold, bow_filter_threshold)
        trainer.run()
        
    if TEST:
        tester = CLWSD2Tester(testdir, outputdir, targetlang,targetwordsfile, sourcetagger, timbloptions, contextsize, DOPOS, DOLEMMAS, bagofwords)
        tester.run()
