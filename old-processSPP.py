#!/usr/bin/python

import os, re, csv, scipy, sklearn, gensim, pandas
from scipy import stats
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

dir = 'ldt'
if dir == 'ldt': t = 'LDT'
elif dir == 'naming': t = 'NT'

z200 = t+'_200ms_Z'
z1200 = t+'_1200ms_Z'
lsa = 'LSA'
w2v = 'w2v'
freq = 'LogSubFreq'

vecmodel = 0
vecmodel = gensim.models.Word2Vec.load(os.path.abspath('/Users/allysonettinger/Desktop/engw2vModel/engModelB'))

  
def testLM():
    x = [10,9,8,7,6]
    y = [2,4,6,8,10]
    mod = sm.OLS(y,x)
    results = mod.fit()
    print results.summary()
    
def analyzeSPP_regr(dir):
    out = open('result_summary.txt','w')
    tgtlexVars = ['LogSubFreq']
    targetLexDF,primeLexDF,relDF,lexDict = readSPP(dir)
    tempDF = relDF[['TargetWord','Prime','Relation1',lsa,z200]]
    for v in tgtlexVars:
        tempDF.insert(len(tempDF.columns.values),v+'_T',np.nan)
        for i in range(len(tempDF.iloc[:,1])):
            tgtWord = tempDF.ix[i,'TargetWord'].lower()
            tempDF.set_value(i,v+'_T',lexDict[tgtWord][v])
    tempDF.insert(len(tempDF.columns.values),'w2v',np.nan)
    for i in range(len(tempDF.iloc[:,1])):
        tgtWord = tempDF.ix[i,'TargetWord'].lower()
        primeWord = tempDF.ix[i,'Prime'].lower()
        try: vecmodel.similarity(tgtWord,primeWord)
        except: sim = np.nan
        else: sim = vecmodel.similarity(tgtWord,primeWord)
        tempDF.set_value(i,'w2v',sim)
    newRelCol = []
    for x in tempDF['Relation1']:
        if type(x) == str:
            x = x.lower()
        if re.match('.*unclas',str(x)):
            newRelCol.append('unclassified')
        elif re.match('.*anton',str(x)):
            newRelCol.append('antonym')
        else: newRelCol.append(x)
    tempDF.insert(len(tempDF.columns.values),'relClean',newRelCol)
    print tempDF['Relation1'].unique()
    print tempDF['relClean'].unique()
    for rel in tempDF['relClean'].unique():
#     constraint = (tempDF[z200].notnull()) & (tempDF[lsa].notnull()) & (tempDF['LogSubFreq_T'].notnull()) & (tempDF['Relation1'] == 'synonym')
        constraint = (tempDF[z200].notnull()) & (tempDF[lsa].notnull()) & (tempDF['w2v'].notnull()) & (tempDF['relClean'] == rel)

    
        try:
            testDF = tempDF[['TargetWord','Prime']][constraint]
            for var in [lsa,z200,w2v]:
                testDF[var] = sklearn.preprocessing.scale(map(float,tempDF[var][constraint]))
        except: continue
        
        tot = len(testDF.iloc[:,0])
            
        for pred in [lsa,w2v]:
            y = testDF[z200]
            X = testDF[pred]
            mod = smf.OLS(y,X)
            results = mod.fit()
            print '\nPREDICTOR: ' + pred + '\n'
            print 'RELATION: ' + rel + '\n'
            print 'TOTAL ITEMS: ' + str(tot) + '\n'
            print results.summary()
    
# def analyzeSPP_regr(dir):
#     lexDict,relDict = readSPP(dir)
#     lex_predictors = []
#     rel_predictors = [lsa]
#     depvar = z200
#     x = []
#     y = []
#     for word in relDict:
#         if vecmodel:
#             try: vecmodel[word]
#             except: continue
#         lp = [lexDict[word][p] for p in lex_predictors]  
#         for prime in relDict[word]:
#             if vecmodel:
#                 try: vecmodel[prime.lower()]
#                 except: continue
#                 else: sim = vecmodel.similarity(word,prime.lower())
#             hasNone = 0
#             rp = [relDict[word][prime][p] for p in rel_predictors] 
# #             rp.append(sim)
#             item_preds = rp+lp
#             for i in item_preds:
#                 if i == None:
# #                     print '\nBAD!!!!!'
# #                     print word
# #                     print ' '.join([str(item) for item in item_preds])
# #                     print '\n'
#                     hasNone = 1
#                     break
#             predicted = relDict[word][prime][depvar]
#             if predicted == None: hasNone = 1
#             if hasNone: continue
# #             print ' '.join([str(item) for item in item_preds])
#             x.append(item_preds)
#             y.append(relDict[word][prime][depvar])
#     
#     x = sklearn.preprocessing.scale(x)
#     y = sklearn.preprocessing.scale(y)
# 
#     mod = sm.OLS(y,x)
#     results = mod.fit()
#     print results.summary()

def analyzeSPP_corr(dir,col1,col2):
    targetLexDF,primeLexDF,relDF,lexDict = readSPP(dir)
    constraint = (relDF[col1].notnull()) & (relDF[col2].notnull())
    v1 = relDF[col1][constraint]
    v2 = relDF[col2][constraint]
    rho,p_rho = scipy.stats.spearmanr(v1,v2)
    r,p = scipy.stats.pearsonr(v1,v2)
    
    print rho
    print p_rho
    print r
    print p

# def analyzeSPP_corr(dir):
#     v2varname = z200
#     lexDict,relDict = readSPP(dir)
#     v1 = []
#     v2 = []
#     for word in relDict:
#         if vecmodel:
#             try: vecmodel[word]
#             except: continue
#         for prime in relDict[word]:
#             if vecmodel:
#                 try: vecmodel[prime.lower()]
#                 except: continue
#                 else: v1var = vecmodel.similarity(word,prime.lower())
#             v1var = relDict[word][prime][lsa]
#             v2var = relDict[word][prime][v2varname]
#             if v1var != None and v2var != None:
#                 v1.append(v1var)
#                 v2.append(v2var)
# #                 print '	'.join([str(x) for x in [word,prime,v1var,v2var]])
#     rho,p_rho = scipy.stats.spearmanr(v1,v2)
#     r,p = scipy.stats.pearsonr(v1,v2)
#     print rho
#     print p_rho
#     print r
#     print p
#     

def readSPP(dir):
    #need to have all the different targets, all their different primes, the lexical characteristics for those primes, and the behavioral 
    #measures -- could also store the priming number to sanity check
    lexDict = {}
    
#     lexList = ['Length','SubFreq','LogSubFreq','LogHAL','ELP '+t+' RT','ELP '+t+' Z','POS']
#     relList = ['Relation1','FAS','BAS','LSA',t+' 200ms RT',t+' 1200ms RT',t+' 200ms Z',t+' 1200ms Z',t+' 200ms Z Priming',t+' 1200ms Z Priming',t+' 200ms RT Priming',t+' 1200ms RT Priming']
    lexList = ['Length','SubFreq','LogSubFreq','LogHAL','ELP_'+t+'_RT','ELP_'+t+'_Z','POS']
    relList2 = [t+'_200ms_Z_Priming',t+'_1200ms_Z_Priming',t+'_200ms_RT_Priming',t+'_1200ms_RT_Priming']
    relList = ['Relation1','LSA','FAS','BAS',t+'_200ms_RT',t+'_1200ms_RT',t+'_200ms_Z',t+'_1200ms_Z','Related','Assoc']
    for file in os.listdir(os.path.abspath(dir)):
        if re.match('assoc_.*\.csv',file): assocRel = os.path.join(dir,file)
        elif re.match('assocunrel_.*\.csv',file): assocUn = os.path.join(dir,file)
        elif re.match('otherassoc_.*\.csv',file): otherAssocRel = os.path.join(dir,file)
        elif re.match('otherassocunrel_.*\.csv',file): otherAssocUn = os.path.join(dir,file)
        elif re.match('target_.*\.csv',file): targetF = os.path.join(dir,file)
    
    frameTGT = pandas.read_csv(targetF)
    l = len(frameTGT['TargetWord'])
    
    frameAR = pandas.read_csv(assocRel)
    frameOAR = pandas.read_csv(otherAssocRel)

    frameAU = pandas.read_csv(assocUn)
    frameOAU = pandas.read_csv(otherAssocUn)
    
    for df in [frameAR,frameOAR]: df['Related'] = 'REL'
    for df in [frameAU,frameOAU]: 
        df['Related'] = 'UNREL'
        df['Relation1'] = 'norel'
        df['FAS'] = None
        df['BAS'] = None
    for df in [frameAR,frameAU]: df['Assoc'] = 'ASSOC'
    for df in [frameOAR,frameOAU]: df['Assoc'] = 'OTHER'
    
    
    [frameTGT,frameAR,frameOAR,frameAU,frameOAU] = colSub([frameTGT,frameAR,frameOAR,frameAU,frameOAU])

    targetLexDF = frameTGT[['TargetWord']+lexList]
    primeLexDF = pandas.merge(frameAR[['Prime']+lexList],frameOAR[['Prime']+lexList], how='outer')
    
    for i in range(len(targetLexDF['TargetWord'])):
        wrd = targetLexDF.ix[i,'TargetWord'].lower()
        lexDict[wrd] = {}
        for it in lexList: lexDict[wrd][it] = targetLexDF.ix[i,it]
    for i in range(len(primeLexDF['Prime'])):
        wrd = primeLexDF.ix[i,'Prime']
        lexDict[wrd] = {}
        for it in lexList: lexDict[wrd][it] = primeLexDF.ix[i,it]
                
    relPreDF1 = pandas.merge(frameAR[['TargetWord','Prime']+relList],frameOAR[['TargetWord','Prime']+relList], how='outer')
    relPreDF2 = pandas.merge(frameAU[['TargetWord','Prime']+relList],frameOAU[['TargetWord','Prime']+relList], how='outer')
#     
    relDF = pandas.merge(relPreDF1,relPreDF2,how='outer')

    
    return targetLexDF,primeLexDF,relDF,lexDict
    
        
def colSub(dfs):
    newdfs = []
    for df in dfs:
        if 'Unrelated' in df.columns.values: df = df.rename(columns={'Unrelated':'Prime'})
        for colName in df.columns.values:
            if re.match('.*\s',colName): df = df.rename(columns={colName:re.sub(' ','_',colName)})
        newdfs.append(df)
    return newdfs
                
#     return lexDict,relDict

# def readSPP_old(dir):
#     #need to have all the different targets, all their different primes, the lexical characteristics for those primes, and the behavioral 
#     #measures -- could also store the priming number to sanity check
#     lexDict = {}
#     relDict = {}
#     
#     lexList = ['Length','SubFreq','LogSubFreq','LogHAL','ELP '+t+' RT','ELP '+t+' Z','POS']
#     relList = ['Relation1','FAS','BAS','LSA',t+' 200ms RT',t+' 1200ms RT',t+' 200ms Z',t+' 1200ms Z',t+' 200ms Z Priming',t+' 1200ms Z Priming',t+' 200ms RT Priming',t+' 1200ms RT Priming']
# 
#     for file in os.listdir(os.path.abspath(dir)):
#         if re.match('assoc_.*\.csv',file): assocRel = os.path.join(dir,file)
#         elif re.match('assocunrel_.*\.csv',file): assocUn = os.path.join(dir,file)
#         elif re.match('otherassoc_.*\.csv',file): otherAssocRel = os.path.join(dir,file)
#         elif re.match('otherassocunrel_.*\.csv',file): otherAssocUn = os.path.join(dir,file)
#         elif re.match('target_.*\.csv',file): targetF = os.path.join(dir,file)
#     
#     with open(targetF) as tgtFile:
#         d = csv.DictReader(tgtFile)
#         for row in d:
#             tgt = row['TargetWord'].lower().strip()
#             lexDict = updateDict(row,lexList,lexDict,tgt)
#             relDict[tgt] = {}
#     for f in [assocRel,otherAssocRel]:
#         with open(f) as file:
#             d = csv.DictReader(file)
#             for row in d:
#                 tgt = row['TargetWord'].lower().strip()
#                 prime = row['Prime'].strip()
#                 lexDict = updateDict(row,lexList,lexDict,prime)
#                 relDict[tgt] = updateDict(row,relList,relDict[tgt],prime)
#     for f in [assocUn,otherAssocUn]:
#         with open(f) as file:
#             d = csv.DictReader(file)
#             for row in d:
#                 tgt = row['TargetWord'].lower().strip()
#                 prime = row['Unrelated'].strip()
#                 relDict[tgt] = updateDict(row,relList,relDict[tgt],prime)
#                 
#     return lexDict,relDict


def updateDict(row,ftList,dictToUpdate,keyToUpdate):
    row = {re.sub('_',' ',c):row[c] for c in row}
    for c in row:
        if len(row[c]) == 0:
            row[c] = None
        elif re.match('-*\.*[0-9]+',row[c]): 
            row[c] = float(row[c])
    for c in ftList:
        if c not in row: row[c] = None
    dictToUpdate[keyToUpdate] = {c:row[c] for c in ftList}
    return dictToUpdate
    
            
if __name__ == "__main__":
#     readSPP(dir)
    analyzeSPP_regr(dir)
#     analyzeSPP_corr(dir,lsa,z200)
#     testLM()