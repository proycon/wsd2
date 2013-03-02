#! /usr/bin/env python
# -*- coding: utf8 -*-

import sys
import campyon
import os
import subprocess
import glob
import shutil
from joblib import Parallel, delayed

WSDDIR = os.path.dirname(os.path.abspath(__file__))

try:
    threads = int(sys.argv[1])
    basedir = sys.argv[2]
    targetwords = sys.argv[3]
    testdir = sys.argv[4]
except:
    print >>sys.stderr,"Usage: exp2.py threads basedir targetwords testdir"
    sys.exit(2)

print >>sys.stderr, "WSDDIR=" + WSDDIR

targetlangs = ['nl','es','it','fr','de']

reference = 'c5lpbR'

contextsizes = [1,2,3,4,5]

def computekeep(c, pos, lemma, bag):
    field = 0
    keep = []
   
    for contextitem in range(0,11): #0 1 2 3 4   5   6 7 8 9 10 
        field += 1
        if contextitem >= 5-c and contextitem <= 5+c :
            keep.append(str(field))
            field += 1
            if pos: keep.append(str(field))
            field += 1
            if lemma: keep.append(str(field))            
        else:
            field +=2
            
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
    global basedir, reference, targetwords, testdir, WSDDIR
    id = computeid(c,pos,lemma,bag)
    print >>sys.stderr,"\nProcessing " + targetlang + " " + id
    print >>sys.stderr,"-------------------------------------------"  
    outputdir = basedir + '/' + targetlang + '/' + id
    try:
        os.mkdir(outputdir)
    except:
        pass
    os.chdir(outputdir)
    
    if not os.path.isdir(basedir + '/' + targetlang + '/' + reference):
        raise Exception("Reference dir does not exist: " + basedir + '/' + targetlang + '/' + reference)
    
    if os.path.exists(outputdir + '/results'): 
        print >>sys.stderr,"Already done, skipping " + id
        return
    
    
    keep = computekeep(c,pos,lemma,bag)
    
    DOTRAIN = False
    for filename in glob.glob(basedir + '/' + targetlang + '/' + reference + '/*.train'):
        outputfile = outputdir + '/' + os.path.basename(filename)        
        if os.path.exists(outputfile):
            basename = outputfile[:-6]
            if not os.path.exists(basename + '.ibase') or not os.path.exists(basename + '.train.paramsearch'):
                DOTRAIN = True

        else:
            DOTRAIN = True
            print >>sys.stderr,"Extracting train files for " + id + " with " + keep + " to " + outputfile   
            extractor = campyon.Campyon('-f',filename, '-o',outputfile,'-k',keep)
            extractor()
        
    if bag:
        for filename in glob.glob(basedir + '/' + targetlang + '/' + reference + '/*.bag'):
            shutil.copyfile(filename,outputdir + '/' + os.path.basename(filename))            
    
    if DOTRAIN:
        cmd = 'python ' + WSDDIR + '/wsd2.py --nogen --train -L ' + targetlang + ' -o ' + outputdir + ' -w ' + targetwords
        cmd += ' -c ' + str(c)
        if pos: cmd += ' -p'
        if lemma: cmd += ' -l'
        if bag: cmd += ' -b'
        cmd += ' -s ' + basedir + '/' + targetlang + '/en.txt'
        cmd += ' -t ' + basedir + '/' + targetlang + '/' + targetlang + '.txt'
        cmd += ' -m ' + basedir + '/' + targetlang + '/phrase-table'  
        cmd += ' --Stagger=file:' + basedir + '/' + targetlang + '/en.tagged'
        if os.path.exists(basedir + '/' + targetlang + '/' + targetlang + '.tagged'): 
            cmd += ' --Ttagger=file:' + basedir + '/' + targetlang + '/' + targetlang + '.tagged'
        cmd += ' >&2 2> ' + outputdir + '/train.log'
        print >>sys.stderr,"Training "+ targetlang + " " + id + ": " + cmd        
        r = subprocess.call(cmd, shell=True)
        if r != 0:
            raise Exception("ERROR: Training " + targetlang + " " + id + " FAILED!")    
        print >>sys.stderr,"Done training"
    
    cmd = 'python ' + WSDDIR + '/wsd2.py --test -L ' + targetlang  + ' -o ' + outputdir + ' -T ' + testdir + ' -w ' + targetwords
    cmd += ' -c ' + str(c)
    if pos: cmd += ' -p'
    if lemma: cmd += ' -l'
    if bag: cmd += ' -b'
    cmd += ' --Stagger=freeling:localhost:1850'
    cmd += ' >&2 2> ' + outputdir + '/test.log'
    print >>sys.stderr,"Testing "+ targetlang + " " + id + ": " + cmd               
    r = subprocess.call(cmd, shell=True)
    if r != 0:
        raise Exception("ERROR: Testing " + targetlang + " " + id + " FAILED!")    
    print >>sys.stderr,"Done testing"          

    os.chdir(WSDDIR)

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

if threads > 1:        
    Parallel(n_jobs=threads)(delayed(compute)(*conf) for conf in configurations)
else:
    for conf in configurations:
        compute(*conf)
