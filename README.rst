==============================================
 WSD2: Cross-Lingual Word Sense Disambiguation 2
==============================================

*submission for SemEval 2013 - Task 10*

*by Maarten van Gompel <proycon@anaproy.nl>*
http://github.com/proycon/wsd2
Centre for Language Studies
Radboud University Nijmegen

The WSD2 system uses a k-NN classifier approach using timbl (IB1). It supports
local context features, global context keyword features (bag of word model),
lemma features and part-of-speech features. Machine learning parameters
can be optimised using paramsearch.

Tools/libraries used:
 * Timbl for Machine learning (http://ilk.uvt.nl/timbl)
 * paramsearch for parameter optimisation (http://ilk.uvt.nl/paramsearch)
 * python-timbl (https://github.com/proycon/python-timbl)
 * pynlpl (https://github.com/proycon/pynlpl)
 * Ucto for tokenisation of all languages (http://ilk.uvt.nl/ucto)
 * Frog for PoS-tagging and Lemmatisation of Dutch (http://ilk.uvt.nl/frog)
 * FreeLing for PoS-tagging and Lemmatisation of English, Spanish, Italian (http://nlp.lsi.upc.edu/freeling/)
 * TreeTagger for Pos-tagging and Lemmatisation of French and German (http://www.ims.uni-stuttgart.de/projekte/corplex/TreeTagger/)
 * GIZA++ for building training data (intersection of alignments)  (http://www.statmt.org/moses/giza/GIZA++.html) (not invoked by system, apply manually)
 * scorer_task3.pl by Diana McCarthy, adapted by Els Lefever, for the Cross-Lingual Lexical Substitution Task SemEval 2010 (included with system)

Test data should be in the XML format as specified by Cross-Lingual Word Sense Disambiguation task for Semeval 2010/2013

Licensed under GNU Public License v3
 
