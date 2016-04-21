#!/bin/bash
#PBS -N gensMiniUW
#PBS -l walltime=24:00:00
#PBS -M aetting@umd.edu
#PBS -m ae
#PBS -l pmem=15gb

dir1=/Users/allysonettinger/Desktop/meaning_cc/priming/corpora
#dir2='/fs/clip-xling/projects/polysemy/bolt-zh-puncproc'
#dir3='/fs/clip-xling/projects/polysemy/onto-zh-puncproc'
modelout=/Users/allysonettinger/Desktop/meaning_cc/priming/testmodels/testlsi


# source virtENV/bin/activate
python LSIWrapper.py $modelout $dir1
# deactivate

