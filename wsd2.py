#! /usr/bin/env python
# -*- coding: utf8 -*-

import sys
import getopt
import os
import codecs
from lxml import etree as ElementTree
from pynlpl.formats.moses import PhraseTable
from pynlpl.formats.giza import GizaModel
from pynlpl.tagger import Tagger
import timbl
import glob
import random
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot

WSDDIR = os.path.dirname(__file__)

def usage():
    """Print usage instructions"""
    print >> sys.stderr,"Usage: wsd2.py --train -L [lang] -s [source-text] -t [target-text] -m [moses-phrasetable] -w [targetwords-file] -o [outputdir] -O [timbloptions]"
    print >> sys.stderr,"       wsd2.py --test -L [lang] -T [testdir] -w [targetwords-file] -o [outputdir] -O [timbloptions]"
    print >> sys.stderr," -c [int]   context size"
    print >> sys.stderr," -l         add lemmatisation features"
    print >> sys.stderr," -p         add PoS-tag features"    
    print >> sys.stderr," -b         Generate Bag-of-Words for keywords global context"
    print >> sys.stderr," -B [absolute_threshold,probability_threshold,filter_threshold]"
    print >> sys.stderr,"                        Parameters for the selection of bag of words."            
    print >> sys.stderr,"          1) Bag-of-word needs to occur at least x times in context"
    print >> sys.stderr,"          2) Bag-of-word needs to have sense|keyword probability of at least x"
    print >> sys.stderr,"          3) Filter out words with a global corpus occurence less than x"
    print >> sys.stderr," -R                     Automatically compute absolute threshold per word-expert"
    print >> sys.stderr," --Stagger   Tagger for source language, set to frog:[port] or freeling:[channel] or corenlp, start the tagger server manually first for the first two"
    print >> sys.stderr," --Ttagger   Tagger for target language, set to frog:[port] or freeling:[channel] (start the tagger server manually first) or  de.lex or fr.lex for built-in lexicons.. "
    print >> sys.stderr," -V          Produce input for voter system (choose a different outputdirectory for each classifier)"    
        
class TestSet(object):
    languages = {
        'english': 'en',
        'french': 'fr',
        u"français": 'fr',
        'italian': 'it',
        'italiano': 'it',
        'german': 'de',
        'deutsch': 'de',
        'dutch': 'nl',
        'nederlands': 'nl',
        'spanish': 'es',
        u"español": 'es',
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
        print >>sys.stderr, "Loading and tokenising " + filename.encode('utf-8')
        tree = ElementTree.parse(filename)
        root = tree.xpath("/corpus")
        if len(root) > 0: 
            root = root[0]
        else:
            raise Exception("This is not a valid test-file!")
        self.lang = TestSet.languages[root.attrib['lang'].lower()]        
        


        for lemmanode in root.findall('.//lexelt'):
            lemma, pos = lemmanode.attrib['item'].rsplit(".",1)

            instances = {}
            tmpfilename = '/tmp/' + "%032x" % random.getrandbits(128)
            tmpfile = codecs.open(tmpfilename,'w','utf-8')
            tmpmap = []
            idmap = []
            
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
                idmap.append(id)
                tmpmap.append( (leftcontext, head, rightcontext) )
                tmpfile.write(leftcontext + head + rightcontext + "\n")
                #instances[id] = (leftcontext,head,rightcontext) #room for output classes is reserved (set to None)
            
            tmpfile.close()
            r = os.system('ucto -L' + self.lang +  ' -m -n ' + tmpfilename + ' ' + tmpfilename +'.out')
            if r != 0:
                raise Exception("ucto failed")
            
            print >>sys.stderr, "Processing ucto output from "+ tmpfilename + '.out'
            f = codecs.open(tmpfilename + '.out','r','utf-8')
            for i, line in enumerate(f):
                if i == len(tmpmap): break
                leftcontext_untok, head, rightcontext_untok = tmpmap[i]
                line = line.strip()
                words = line.split(' ')
                
                origindex = len(leftcontext_untok.split(' '))
                mindistance = 9999
                focusindex = -1
                
                for j, word in enumerate(words): 
                    if word == head:
                        distance = abs(origindex - j)
                        if distance <= mindistance:
                            focusindex = j
                            mindistance = distance                 
                                                                                           
                if focusindex  == -1:
                    print >>sys.stderr,"Full match not found, attempting to find partial match" 
                    #final partial match:
                    for j, word in enumerate(words): 
                        partialfound = word.find(head)
                        if partialfound != -1:
                            distance = abs(origindex - j)
                            if distance <= mindistance:
                                focusindex = j
                                mindistance = distance
                    
                    if focusindex != -1:
                        leftcontext = u" ".join(words[:focusindex])
                        if partialfound > 0:
                            leftcontext += " " + words[focusindex][:partialfound]             
                        rightcontext = u" ".join(words[focusindex + 1:])
                        
                        
                        if words[focusindex][partialfound + len(head):]:
                            rightcontext = words[focusindex][partialfound + len(head):] + ' ' + rightcontext 
                    else:            
                        raise Exception("Focus word not found after tokenisation! This should not happen! head=" + head.encode('utf-8') + ",words=" + ' '.join(words).encode('utf-8'))
                else:
                    leftcontext = u" ".join(words[:focusindex])
                    rightcontext = u" ".join(words[focusindex + 1:])

                                               
                instances[idmap[i]] = (leftcontext, head, rightcontext)
                
            f.close()
            
            try:
                os.unlink(tmpfilename)
                os.unlink(tmpfilename + '.out')
            except:
                pass

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
        line = line.lstrip( unicode( codecs.BOM_UTF8, "utf-8" ) )
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
    
    def __init__(self, outputdir, targetlang, phrasetable, gizamodel_s2t, gizamodel_t2s, sourcefile, targetfile, targetwordsfile, sourcetagger, targettagger, contextsize, DOPOS, DOLEMMAS, DOVOTER, exemplarweights, timbloptions, bagofwords, compute_bow_params, bow_absolute_threshold, bow_prob_threshold, bow_filter_threshold, maxdivergencefrombest = 0.5):      
        if phrasetablefile and not os.path.exists(phrasetablefile):
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
        self.phrasetable = phrasetable
        self.gizamodel_s2t = gizamodel_s2t
        self.gizamodel_t2s = gizamodel_t2s
        
        self.contextsize = contextsize
        self.DOPOS = DOPOS
        self.DOLEMMAS = DOLEMMAS
        self.DOVOTER = DOVOTER
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
        if not (focuslemma,focuspos) in count:
            print "focusword not seen", focuslemma.encode('utf-8') +'.'+ focuspos
            return 0 #focus word has not been counted for

        if not senselabel in count[(focuslemma,focuspos)]:
            print "sense not seen:", senselabel.encode('utf-8')
            return 0 #sense has never been seen for this focus word

        if not (lemma,pos) in totalcount:
            print "keyword not seen:", lemma.encode('utf-8')+'.'+pos
            return 0 #keyword has never been seen

        Ns_kloc = 0.0
        if (lemma,pos) in count[(focuslemma,focuspos)][senselabel]:
            Ns_kloc = float(count[(focuslemma,focuspos)][senselabel][(lemma,pos)])        

        Nkloc = 0
        for sense in count[(focuslemma,focuspos)]:
            if (lemma,pos) in count[(focuslemma,focuspos)][sense]:
                Nkloc += count[(focuslemma,focuspos)][sense][(lemma,pos)]
                

        Nkcorp = float(totalcount[(lemma,pos)]) #/ float(totalcount_sum)

        #if focuslemma == 'wild':
        #    print "p = (",Ns_kloc,"/",Nkloc,") * 1/",Nkcorp, " = ",  (Ns_kloc / Nkloc) * (1/Nkcorp)

        return (Ns_kloc / float(Nkloc)) * (1/Nkcorp)


    def make_bag_of_words(self, focuslemma, focuspos, bow_absolute_threshold, count, totalcount):
        print >>sys.stderr, "Computing and writing bag for " + focuslemma.encode('utf-8') + "..."

        if not (focuslemma,focuspos) in count:
            print >>sys.stderr, "   WARNING: No count found!"
            return [] #focus word has not been counted for

        bag = []
        #select all words that occur at least 3 times for a sense, and have a probability_sense_given_keyword >= 0.001
        for sense in count[(focuslemma,focuspos)]:
            for lemma, pos in count[(focuslemma,focuspos)][sense].keys():
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

        print >>sys.stderr, "\tFound " + str(len(bag)) + " keywords"
        return bag


        

    def run(self):        

        count = {}
        totalcount = {}
        totalcount_sum = 0
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
            
            if self.gizamodel_s2t:
                iter_s2t = iter(self.gizamodel_s2t)
                iter_t2s = iter(self.gizamodel_t2s)
                
            
            for sentencenum, (sourceline, targetline) in enumerate(zip(f_source, f_target)):                    
                print >>sys.stderr, "@" + str(sentencenum+1)            
                
                if self.gizamodel_s2t: 
                    try:
                        s2t = iter_s2t.next() #self.gizamodel_s2t.next()
                        t2s = iter_t2s.next() #self.gizamodel_t2s.next()
                    except StopIteration:
                        print >>sys.stderr,"WARNING: No more GIZA alignments, breaking"
                        break
                    #print >>sys.stderr, "S2T: " + repr(s2t)
                    #print >>sys.stderr, "T2S: " + repr(t2s)
                    intersection = s2t.intersect(t2s) 
                    #print >>sys.stderr, "INT: " + repr(intersection)
                else: 
                    intersection = None
                
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
                    if self.bagofwords and not finalstage:
                        if not (sourcelemma, sourcepos) in totalcount:                 
                            totalcount[(sourcelemma, sourcepos)] = 1
                        else:
                            totalcount[(sourcelemma, sourcepos)] += 1
                        totalcount_sum += 1
                    
                    
                    if (sourcelemma, sourcepos) in self.targetwords:
                        print >>sys.stderr, " @" + str(sentencenum+1) + ":" + str(i) + " -- Found " + sourcelemma.encode('utf-8') + '.' + sourcepos,
                        target = None
                        Pst = Pts = 0
                        if intersection != None:
                            target, foundindex = intersection.getalignedtarget(i)
                            if isinstance(foundindex, tuple):
                                targetl = foundindex[1]
                                foundindex = foundindex[0]
                            
                        
                        #Is this sourceword aligned? 
                        if (self.phrasetable != None and sourceword in self.phrasetable) or (intersection != None and target != None):
                                                
                            #find options in phrasetable
                            if self.phrasetable:
                                try:
                                    translationoptions = self.phrasetable[sourceword]  #[ (target, Pst, Pts, null_alignments) ]
                                except KeyError:
                                    continue
                            elif self.gizamodel_s2t:                        
                                #We already have the aligned target
                                translationoptions = None
                                print >>sys.stderr, " aligned with '" + target.encode('utf-8') + "'"
                            
                            
                            
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
                            

                            if translationoptions:                        
                                #Find which translation option is the best match here, only one may be used
                                bestpossiblescore = max([ x[2] for x in translationoptions])
                                bestscore = 0
                                best = None                        
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
                                        if Pts > bestscore: 
                                            bestscore = Pts
                                            best = (target, Pts, foundindex)
                                
                                if not best:
                                    print >>sys.stderr,"No translation options match"
                                    continue
                                elif bestscore < bestpossiblescore * self.maxdivergencefrombest:
                                    print >>sys.stderr,"Matching translation option '" + target + "' scores too low (" + str(bestscore) + " vs " + str(bestpossiblescore) + ")"
                                    continue 
                                else:                                    
                                    target, Pts, foundindex = best
                                    print >>sys.stderr, " aligned with '" + target.encode('utf-8') + "'"
                                                
                            #get lemmatised form of target word
                            if self.targettagger:
                                if ' ' in target:
                                    if self.phrasetable:
                                        target = ' '.join(targetlemmas[foundindex:foundindex+len(targetl)])
                                    else:
                                        target = ' '.join(targetlemmas[foundindex:foundindex+targetl])
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
                                if not (sourcelemma,sourcepos) in count:
                                    count[(sourcelemma,sourcepos)] = {}
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
                print >>sys.stderr, "Making bags of words"
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
        if self.DOVOTER:
            self.makevoterinput()
                
    def loadclassifiers(self):            
        for lemma,pos in self.targetwords:            
            print >>sys.stderr, "Loading " + lemma.encode('utf-8')

            if os.path.exists(self.outputdir + '/' + lemma +'.' + pos + '.' + self.targetlang + '.train'):
                self.classifiers[(lemma,pos,self.targetlang)] = timbl.TimblClassifier(self.outputdir + '/' + lemma +'.' + pos + '.' + self.targetlang, timbloptions)        

    def run2(self):        
        print >>sys.stderr, "Training " + str(len(self.classifiers)) + " classifiers"
        for classifier in self.classifiers:
            self.classifiers[classifier].train()
            self.classifiers[classifier].save()
            
        print >>sys.stderr, "Parameter optimisation"
        for f in glob.glob(self.outputdir + '/*.train'):
            os.system("paramsearch ib1 " + f + " > " + f + ".paramsearch")


    def makevoterinput(self):
        """Make traindata for voter by testing on traindata"""
        print >>sys.stderr, "Generating voter input by classifying traindata"    
        for classifier in self.classifiers:
            print >>sys.stderr, "Making voter input for " + classifier.fileprefix.encode('utf-8') + '.votertrain'
            f_out = codecs.open(classifier.fileprefix + '.votertrain','w','utf-8')
            f_in = codecs.open(classifier.fileprefix + '.train','r','utf-8')
            for line in f_in:
                line = line.strip()
                fields = line.split("\t")
                features = fields[:-1]
                classlabel, distribution, distance =  self.classifiers[classifier].classify(features)
                f_out.write(classlabel + "\t" + fields[-1])
            f_in.close()
            f_out.close()
            
        
        
def paramsearch2timblargs(filename):
    opts = ""
    f = open(filename)
    lines = f.readlines()
    f.close()
    if lines:
        line = lines[-1].strip()
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
    return opts

def processresult(out_best, out_oof, id, lemma, pos, targetlang, bestsense, distribution, distance)     
    bestscore = max(distribution.values())
    bestsenses = [ sense for sense, score in distribution.items() if score == bestscore ]
    fivebestsenses = [ sense for sense, score in sorted(distribution.items()[:5], key=lambda x: -1 * x[1]) ]
    bestsenses_s = ';'.join(bestsenses)
    fivebestsenses_s = ';'.join(fivebestsenses)
    if not isinstance(bestsenses_s,unicode): bestsenses_s  = unicode(bestsenses_s,'utf-8')
    if not isinstance(fivebestsenses_s,unicode): fivebestsenses_s  = unicode(fivebestsenses_s,'utf-8')
    out_best.write(lemma + "." + pos + "." + targetlang + ' ' + str(id) + ' :: ' + bestsenses_s + ';\n')
    out_oof.write(lemma + "." + pos + "." + targetlang + ' ' + str(id) + ' ::: ' + fivebestsenses_s + ';\n')
    print >>sys.stderr, "<-- Timbl output for " + lemma.encode('utf-8') + '.' + pos + " @" + str(id) + ": " + repr(distribution)


    
class CLWSD2Tester(object):          
    def __init__(self, testdir, outputdir, targetlang,targetwordsfile, sourcetagger, timbloptions, contextsize, DOPOS, DOLEMMAS, bagofwords, DOVOTER):        
        self.sourcetagger = sourcetagger
        
        
        print >>sys.stderr, "Loading Target Words " + targetwordsfile       
        self.targetwords = loadtargetwords(targetwordsfile)
        self.classifiers = {}
        
        self.targetlang = targetlang
        
        self.contextsize = contextsize
        self.DOPOS = DOPOS
        self.DOLEMMAS = DOLEMMAS
        self.DOVOTER = DOVOTER
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
        self.bags = {}        
        if self.bagofwords:
            #load bags
            for bagfile in glob.glob(outputdir + "/*.bag"):
                print >>sys.stderr, "Loading bag " + bagfile
                focuslemma,focuspos,_ = os.path.basename(bagfile).split(".")
                focuslemma = unicode(focuslemma,'utf-8')
                self.bags[(focuslemma,focuspos)] = []
                f = codecs.open(bagfile,'r','utf-8')
                for line in f:
                    fields = line.split("\t")
                    if not (fields[0],fields[1]) in self.bags[(focuslemma,focuspos)]:
                        self.bags[(focuslemma,focuspos)].append((fields[0],fields[1]))
                f.close()
              
       
    def run(self):
        global WSDDIR
        print >>sys.stderr, "Extracting features from testset"
        for lemma,pos in self.testset.lemmas():            
            print >>sys.stderr, "Processing " + lemma.encode('utf-8')

            timbloptions = self.timbloptions 
            if os.path.exists(self.outputdir + '/' + lemma +'.' + pos + '.' + self.targetlang + '.train.paramsearch'):
                o = paramsearch2timblargs(self.outputdir + '/' + lemma +'.' + pos + '.' + self.targetlang + '.train.paramsearch')
                timbloptions += " " + o 
                print >>sys.stderr, "Parameter optimisation loaded: " + o
            else:            
                print >>sys.stderr, "NOTICE: No parameter optimisation found!"
            
            print >>sys.stderr, "Instantiating classifier " + lemma.encode('utf-8') + " with options: " + timbloptions
            classifier = timbl.TimblClassifier(self.outputdir + '/' + lemma +'.' + pos + '.' + self.targetlang, timbloptions)
            out_best = codecs.open(self.outputdir + '/' + lemma + '.' + pos + '.best','w','utf-8')
            out_oof = codecs.open(self.outputdir + '/' + lemma + '.' + pos + '.oof','w','utf-8')
            if self.DOVOTER:
                out_votertest =  codecs.open(self.outputdir + '/' + lemma + '.' + pos + '.votertest','w','utf-8')
                
            for instancenum, (id, ( leftcontext,head,rightcontext)) in enumerate(self.testset.instances(lemma,pos)):
                print >>sys.stderr, "--> " + lemma.encode('utf-8') + '.' + pos + " @" + str(instancenum+1) + ": " + leftcontext.encode('utf-8') + " *" + head.encode('utf-8') + "* " + rightcontext.encode('utf-8')
                
                sourcewords_pretagged = leftcontext + ' ' + head + ' ' + rightcontext
                
                sourcewords, sourcepostags, sourcelemmas = sourcetagger.process(sourcewords_pretagged.split(' '))
                sourcepostags = [ x[0].lower() for x in sourcepostags ]
                
                              
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
                    raise Exception("Focus word not found after tagging! This should not happen! head=" + head.encode('utf-8') + ",words=" + ' '.join(sourcewords).encode('utf-8'))
                         
            
                sourcelemma = sourcelemmas[focusindex]
                sourcepos = sourcepostags[focusindex]                         
                         
                #grab local context features
                features = []                    
                for j in range(focusindex - self.contextsize, focusindex + 1 + self.contextsize):
                    if j > 0 and j < len(sourcewords):
                        features.append(sourcewords[j])
                        if self.DOPOS:
                            if sourcepostags[j]:
                                features.append(sourcepostags[j])
                            else:
                                features.append("?")
                        if self.DOLEMMAS:
                            if sourcelemmas[j]: 
                                features.append(sourcelemmas[j])
                            else:
                                features.append("?")
                    else:
                        features.append("{NULL}")
                        if self.DOPOS: features.append("{NULL}")
                        if self.DOLEMMAS: features.append("{NULL}")     
                        
                if self.bagofwords:                    
                    if (sourcelemma,sourcepos) in self.bags:
                        for keylemma,keypos in self.bags[(sourcelemma,sourcepos)]:
                            found = False
                            for j in range(focusindex - self.contextsize, focusindex + 1 + self.contextsize):
                                if j > 0 and j < len(sourcewords) and j != focusindex:
                                    if sourcelemmas[j] == keylemma and sourcepostags[j] == keypos: 
                                        found = True
                                        break
                            
                            #Write bag-of-word features
                            if found:
                                features.append("1")
                            else:
                                features.append("0")                          
                    else:
                        print >>sys.stderr, 'NOTICE: ' + sourcelemma.encode('utf-8')+ ' ' + sourcepos + ' has no bag'
                    

                print " -- Classifier features: " + repr(features)                                        
                bestsense, distribution, distance = classifier.classify(features)
                
                processresult(out_best, out_oof, id, lemma, pos, targetlang, bestsense, distribution, distance)
                
                if self.DOVOTER: out_votertest.write(str(id) + "\t" + sourcewords[focusindex]+ "\t"+ bestsense + "\n")
                
            out_best.close()
            out_oof.close()
            if DOVOTER:
                out_votertest.close()
            
            #score
            os.system('perl ' + WSDDIR + '/ScorerTask3.pl ' + outputdir + '/' + lemma + '.' + pos + '.best' + ' ' + WSDDIR + '/data/trial/' + self.targetlang + '/' + lemma + '_gold.txt 2> ' + outputdir + '/' + lemma + '.' + pos + '.best.scorerr')
            os.system('perl ' + WSDDIR + '/ScorerTask3.pl ' + outputdir + '/' + lemma + '.' + pos + '.oof' + ' ' + WSDDIR + '/data/trial/' + self.targetlang + '/' + lemma + '_gold.txt -t oof 2> ' + outputdir + '/' + lemma + '.' + pos + '.oof.scorerr')
            
        scorereport(outputdir)
                 
def scorereport(self, outputdir):
    f = codecs.open(outputdir + '/results','w','utf-8')
    f.write('BEST RESULTS\n-------------\n')
    
    rlist = []
    plist = []
    
    for filename in glob.glob(outputdir + '/*.best.results'):
        lemma,pos = os.path.basename(filename).split('.')[:2]
        f_in = open(filename,'r')
        for line in f_in:
            if line[:12] == "precision = ":                    
                p = float(line[12:line.find(',')] )
                r =  float(line[line.find('recall = ') + 9:] )
                plist.append( p )
                rlist.append( r )
                f.write(lemma + ":\t" + str(p) + "\t" + str(r) + "\n")
                break 
        f_in.close()
        
    f.write("AVERAGE:\t" + str(sum(plist) / float(len(plist))) + "\t" + str(sum(rlist) / float(len(rlist)))+"\n")


    rlist = []
    plist = []

    f.write('\n\nOUT OF FIVE RESULTS\n-------------\n')
    for filename in glob.glob(outputdir + '/*.oof.results'):
        lemma,pos = os.path.basename(filename).split('.')[:2]
        f_in = open(filename,'r')
        for line in f_in:
            if line[:12] == "precision = ":                    
                p = float(line[12:line.find(',')] )
                r =  float(line[line.find('recall = ') + 9:] )
                plist.append( p )
                rlist.append( r )
                f.write(lemma + ":\t" + str(p) + "\t" + str(r) + "\n")
                break 
        f_in.close()
        
    f.write("AVERAGE:\t" + str(sum(plist) / float(len(plist))) + "\t" + str(sum(rlist) / float(len(rlist)))+"\n")
                    
        
        
    f.close()
    os.system("cat " + outputdir + '/results')
        
        
if __name__ == "__main__":
    try:
	    opts, args = getopt.getopt(sys.argv[1:], "a:s:t:c:lpbB:Ro:w:L:O:m:T:V", ["train","test", "nogen", "Stagger=","Ttagger="])
    except getopt.GetoptError, err:
	    # print help information and exit:
	    print str(err)
	    usage()
	    sys.exit(2)           
    
    TRAIN = TEST = False
    TRAINGEN = True
    sourcefile = targetfile = phrasetablefile = ""
    targetwordsfile = WSDDIR + "/data/targetwords.trial"
    DOLEMMAS = False
    DOPOS = False
    DOVOTER = False
    sourcetagger = None
    targettagger = None
    outputdir = "."
    testdir = WSDDIR + "/data/trial"
    targetlang = ""
    exemplarweights = False
    timbloptions = "-a 0 -k 1"
    contextsize = 0
    
    gizafile_s2t = ""
    gizafile_t2s = ""
    gizamodel_s2t = None
    gizamodel_t2s = None
    
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
        elif o == "--nogen":	
            TRAINGEN = False  
        elif o == '-a':                      
            fields = a.split(':')
            gizafile_s2t = fields[0]
            gizafile_t2s = fields[1]
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
        elif o == '-V':
            DOVOTER = True
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
        
        if not phrasetablefile and not (gizafile_s2t and gizafile_t2s):
            print >>sys.stderr, "ERROR: No phrasetable or giza models specified"
            sys.exit(2)
            
        elif not targettagger:            
            print >>sys.stderr, "WARNING: No target tagger specified"
        
        if TRAINGEN and phrasetablefile:
            print >>sys.stderr, "Loading phrasetable..."
            phrasetable = PhraseTable(phrasetablefile, False, False, "|||", 5, 4, 1)
        else:
            phrasetable = None    
        
        if TRAINGEN and gizafile_s2t and gizafile_t2s:
            print >>sys.stderr, "Loading GIZA model s->t..."
            gizamodel_s2t = GizaModel(gizafile_s2t)        
            print >>sys.stderr, "Loading GIZA model t->s.."
            gizamodel_t2s = GizaModel(gizafile_t2s)
        else:
            gizamodel_s2t = None
            gizamodel_t2s = None
                   
        trainer = CLWSD2Trainer(outputdir, targetlang, phrasetable, gizamodel_s2t, gizamodel_t2s, sourcefile, targetfile, targetwordsfile, sourcetagger, targettagger, contextsize, DOPOS, DOLEMMAS, DOVOTER, exemplarweights, timbloptions, bagofwords,compute_bow_params, bow_absolute_threshold, bow_prob_threshold, bow_filter_threshold)
        if not TRAINGEN:
            trainer.loadclassifiers()
            trainer.run2()
        else:
            trainer.run()
        
    if TEST:
        tester = CLWSD2Tester(testdir, outputdir, targetlang,targetwordsfile, sourcetagger, timbloptions, contextsize, DOPOS, DOLEMMAS, bagofwords, DOVOTER)
        tester.run()
