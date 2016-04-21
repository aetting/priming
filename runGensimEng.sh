#!/bin/bash
#PBS -N gensMiniUW
#PBS -l walltime=24:00:00
#PBS -M aetting@umd.edu
#PBS -m ae
#PBS -l pmem=15gb

#dir1=/fs/clip-xling/projects/polysemy/ukwac-mini/
#dir2='/fs/clip-xling/projects/polysemy/bolt-zh-puncproc'
#dir3='/fs/clip-xling/projects/polysemy/onto-zh-puncproc'
dirlocal=/Users/allysonettinger/Desktop/meaning_cc/priming/corpora
modelout=/Users/allysonettinger/Desktop/meaning_cc/priming/testmodels/testw2v


#source virtENV/bin/activate
python gensimWrapper.py $modelout $dirlocal
#deactivate

