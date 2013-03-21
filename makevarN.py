#! /usr/bin/env python
# -*- coding: utf8 -*-

import sys
import glob
import os
import shutil

dir = sys.argv[1]
targetlangs = ('es','fr','de','it','nl')

for targetlang in targetlangs:
    for d in glob.glob(dir + '/' + targetlang + '/c*'):
        if os.path.isdir(d) and d[-1] != 'N':
            try:
                shutil.copytree(d, d + 'N')            
                os.system("rm " + d + 'N/*.paramsearch')
                os.system("rm " + d + 'N/*.bestsetting')
            except:
                pass
            os.chdir(d+'N')
            confname = os.path.basename(d)
            if '0' in confname:
                c = 0
            elif '1' in confname:
                c = 1                
            elif '2' in confname:
                c = 2                
            elif '3' in confname:
                c = 3
            elif '4' in confname:
                c = 4
            elif '5' in confname:
                c = 5                                                
            options = ""
            if 'l' in confname:
                options += ' -l'
            if 'p' in confname:
                options += ' -p'
            if 'b' in confname:
                options += ' -b'
            os.system("python ~/wsd2/wsd2.py --test -T /home/proycon/wsd2/data/trial -L " + targetlang + " -c " + str(c) + " " + options + " -s ../en.txt -t ../" + targetlang + ".txt -a ../"+targetlang+"-en.A3.final:../en-" + targetlang + ".A3.final  --Stagger=freeling:localhost:1850")
                
            



