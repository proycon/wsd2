#! /usr/bin/env python
# -*- coding: utf8 -*-

import sys
import campyon
import os
import glob
from joblib import Parallel, delayed

try:
    threads = int(sys.argv[1])
    basedir = sys.argv[2]
    targetwords = sys.argv[3]
    testdir = sys.argv[4]
except:
    print >>sys.stderr,"Usage: exp.py threads basedir targetwords testdir"
    sys.exit(2)

targetlangs = ['nl','es','it','fr','de']

reference = 'c5lpbR'

contextsizes = [1,2,3,4,5]

def computekeep(c, pos, lemma, bag):
    field = 0
    keep = []
    contextitem = 0
    
    while True: 
        field += 1
        contextitem += 1
        if contextitem > c:
            break
        keep.append(field)
        field += 1
        if pos: keep.append(field)
        field += 1
        if lemma: keep.append(field)        

    if bag:         
        if field == 0: field = 1
        keep.append(str(field) + ':-1')
    else:
        keep.append('-1')
    return ','.join(keep)      
    
def computeid(c,pos,lemma,bag):
    s =  'c' + str(c)
    if lemma: s += 'l'
    if pos: s += 'p'
    if bag: s += 'b'
    return s
    
    
def compute(targetlang, c,pos,lemma,bag):
    global basedir, reference, targetwords, testdir
    id = computeid(c,pos,lemma,bag)
    print >>sys.stderr,"Processing " + targetlang + " " + id 
    outputdir = basedir + '/' + targetlang + '/' + id
    try:
        os.mkdir(outputdir)
    except:
        pass
    
    if not os.path.isdir(basedir + '/' + targetlang + '/' + reference):
        raise Exception("Reference dir does not exist: " + basedir + '/' + targetlang + '/' + reference)
        
    
    for filename in glob.glob(basedir + '/' + targetlang + '/' + reference + '/*.train'):
        outputfile = outputdir + '/' + os.path.basename(filename)
        print >>sys.stderr,"Extracting train files"
        extractor = campyon.Campyon('-f',filename, '-o',outputfile,'-k',computekeep(c,pos,lemma,bag))
        extractor.run()
        
        cmd = 'python wsd2.py --nogen --train -L ' + targetlang + ' -o ' + outputdir + ' -w ' + targetwords
        cmd += ' -c ' + str(c)
        if pos: cmd += ' -p'
        if lemma: cmd += ' -p'
        if bag: cmd += ' -b'
        cmd += ' -s ' + basedir + '/' + targetlang + '/en.txt'
        cmd += ' -t ' + basedir + '/' + targetlang + '/' + targetlang + '.txt'
        cmd += ' -a ' + basedir + '/' + targetlang + '/' + targetlang + '-' + 'en.A3.final:' + basedir + '/' + targetlang + '/en-' + targetlang + '.A3.final'  
        cmd += ' --Stagger=file:' + basedir + '/en.tagged'
        if os.path.exists(basedir + '/' + targetlang + '.tagged'): 
            cmd += ' --Ttagger=file:' + basedir + '/' + targetlang + '.tagged'
        print >>sys.stderr,"Training system: " + cmd        
        r = os.system(cmd)
        if r != 0:
            print >>sys.stderr,"ERROR: Training " + targetlang + " " + id + " FAILED!"    
        
        cmd = 'python wsd2.py --nogen --train -L ' + targetlang  + ' -o ' + outputdir + ' -T ' + testdir + ' -w ' + targetwords
        cmd += ' -c ' + str(c)
        if pos: cmd += ' -p'
        if lemma: cmd += ' -p'
        if bag: cmd += ' -b'
        cmd += ' --Stagger=freeling:localhost:1850'
        print >>sys.stderr,"Testing system: " + cmd       
        r = os.system(cmd)
        if r != 0:
            print >>sys.stderr,"ERROR: Testing " + targetlang + " " + id + " FAILED!"    
                

configurations = []
for targetlang in targetlangs:
    for c in contextsizes:
        configurations.append( (targetlang, c,False,False,False) ) 
        configurations.append(  (targetlang, c,True,False,False) )
        configurations.append(  (targetlang, c,False,True,False)  )
        configurations.append(  (targetlang, c,False,False,True)  )       
        configurations.append(  (targetlang, c,True,True,False) )
        configurations.append(  (targetlang, c,False,True,True) )
        configurations.append(  (targetlang, c,True,False,True) )
        configurations.append(  (targetlang, c,True,True,True) )                    
        
Parallel(n_jobs=threads)(delayed(compute)(*conf) for conf in configurations)