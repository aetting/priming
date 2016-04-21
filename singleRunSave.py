#!/usr/bin/python

import processSPP
from processSPP import *

# w2v_small = gensim.models.Word2Vec.load(os.path.abspath('/Users/allysonettinger/Desktop/engw2vModel/enModel'))
# w2v_uk1 = gensim.models.Word2Vec.load(os.path.abspath('/Users/allysonettinger/Desktop/engw2vModel/ukWAC-mini-lower-10min'))
# 
# gloveWG100 = readVectors('/Users/allysonettinger/Desktop/glove/glove-Wik-Gig/glove.6B.100d.txt')
# gloveTW100 = readVectors('/Users/allysonettinger/Desktop/glove/glove-Twitter/glove.twitter.27B.100d.txt')
# 
# w2v_big = gensim.models.Word2Vec.load(os.path.abspath('/Users/allysonettinger/Desktop/engw2vModel/engModelB'))
# w2v_ukfull = gensim.models.Word2Vec.load(os.path.abspath('/Users/allysonettinger/Desktop/engw2vModel/ukWAC-full-lower-10min'))
# w2v15_WG = gensim.models.Word2Vec.load(os.path.abspath('/Users/allysonettinger/Desktop/engw2vModel/w2v-15-WG'))

w2v_WG15=readVectors('/fs/clip-cognitive/models/WG15-w2v.txt.gz')
print 'ONE'
w2v_WG5=readVectors('/fs/clip-cognitive/models/WG5-w2v.txt.gz')
print 'TWO'
Gl_WG15=readVectors('/fs/clip-cognitive/models/WG15-Gl.txt.gz')
print 'THREE'
Gl_WG5=readVectors('/fs/clip-cognitive/models/WG5-Gl.txt.gz')
print 'FOUR'

# modelNames = ['w2v_small']
# modelNames = ['w2v_uk1','w2v_big','w2v_ukfull','gloveWG100','gloveTW100']
modelNames = ['w2v_WG15','w2v_WG5','Gl_WG15','Gl_WG5']
models = [eval(e) for e in modelNames]

simdir = '/fs/clip-cognitive/similarity-sets'
simFileNames = [os.path.join(simdir,e) for e in os.listdir(simdir) if re.match('\w',e)]
# 
# syndir = '/Users/allysonettinger/Desktop/meaning_cc/priming/synonymy-sets'
# synFileNames = [os.path.join(syndir,e) for e in os.listdir(syndir) if re.match('\w',e)]


#either create a new CSV from scratch based on SPP files, or update an existing one
#compute similarity or NR values for pairs and output raw (just pairs) and paired (tgt-rel-unrel) DFs


def saveDF(dist,outPrefix,updatePrefix = None):
    
    targetLexDF,primeLexDF,rawDF,pairedDF,lexDict,relDict = readSPP('ldt','LDT')
    if not updatePrefix:
        rawModDF = rawDF[['TargetWord','Prime']]
        pairModDF = pairedDF[['TargetWord','Prime','Unrelated']]
    
    else:
        rawModDF = pandas.read_csv(updatePrefix+'-raw.csv')
        pairModDF= pandas.read_csv(updatePrefix+'-paired.csv')
        
#     relDict = {k:relDict[k] for k in ['abandon','ability','abnormal','above']}
    
    
    if dist == 'nr':
        suf = '_NR'
    elif dist == 'sim':
        suf = '_SIM'
        
    print 'starting models'
    for m in range(len(models)):
        model = models[m]
        modelName = modelNames[m]
        distDict = {}
        for tgt in relDict:
            primes = relDict[tgt].values()
            try: model[tgt]
            except:
                distDict[tgt] = {primes[i]:np.nan for i in range(len(primes))}
                continue
            if dist == 'nr':
                prRanks = []
                for pr in primes:
                    r = getNR(pr,[tgt],model)
                    prRanks.append(r[0])
                distDict[tgt] = {primes[i]:prRanks[i] for i in range(len(primes))}
            elif dist == 'sim':
                distDict[tgt] = {pr:cosSim(tgt,pr,model) for pr in primes}
        rawModDF = addFromDict(rawModDF,'TargetWord','Prime',distDict,modelName+suf)
        pairModDF= addFromDict(pairModDF,'TargetWord','Prime',distDict,modelName+suf+'_rel')
        pairModDF= addFromDict(pairModDF,'TargetWord','Unrelated',distDict,modelName+suf+'_un')
    rawModDF.to_csv(outPrefix+'-raw.csv')
    pairModDF.to_csv(outPrefix+'-paired.csv')


def addFromDict(dfToUpdate,tgtColName,prColName,sourceDict,newColName):
    dfToUpdate.insert(len(dfToUpdate.columns.values),newColName,np.nan)
    for i in range(len(dfToUpdate.iloc[:,1])):
        tgt = dfToUpdate.ix[i,tgtColName]
        pr = dfToUpdate.ix[i,prColName]
        try: dfToUpdate.set_value(i,newColName,sourceDict[tgt][pr])
        except: 
            dfToUpdate.set_value(i,newColName,np.nan)
    return dfToUpdate
    
def getSimSyn():
    with open('texdocs/evalSimSyn.tex','w') as out:
        out.write(r'\documentclass{article}'+'\n')
        out.write(r'\usepackage{multirow}'+'\n')
        out.write(r'\usepackage[margin=.9in]{geometry}'+'\n')
        out.write(r'\begin{document}'+'\n')
        simSetEval(simFileNames,models,modelNames,texTableDoc = out)
        synSetEval(synFileNames,models,modelNames,texTableDoc = out)
        out.write(r'\end{document}')

def saveSimSyn():
    for simFileName in simFileNames:
        simDF = pandas.read_csv(simFileName,delimiter = '\s',engine='python',header=None,names=['w1','w2','SimRating'])
        simDF['w1'] = simDF['w1'].apply(lambda x: re.sub('-.*','',x.lower()))
        simDF['w2'] = simDF['w2'].apply(lambda x: re.sub('-.*','',x.lower()))
#         print simDF
        simDF = addSimCols(simDF,models,modelNames,'w1','w2')
    
        setName = re.match('.+/([^/]+)$',simFileName).group(1)
        simDF.to_csv('simDF-%s.csv'%setName)
            
        

# def readInput(argv):
#     opts, _ = getopt.getopt(sys.argv[1:], "d:o:u:g:t:", ["dist=","outpref=","updatepref=","gensim=","text="])
# 
#     for option, value in opts:
#         if option in ("-d", "--dist"):
#             vecList.append(value)
#         if option in ("-o", "--outpref"):
#             vecList.append(value)
#         if option in ("-d", "--dist"):
#             vecList.append(value)
#         if option in ("-d", "--dist"):
#             vecList.append(value)
#         if option in ("-d", "--dist"):
#             vecList.append(value)
            
if __name__ == "__main__":
    saveDF('sim','simDF',updatePrefix = 'simDF')
#     saveSimSyn()