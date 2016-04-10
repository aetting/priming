#!/usr/bin/python

import os, sys, re, csv, scipy, sklearn, gensim, pandas, math, processSPP_util
from scipy import stats,spatial
from sklearn import preprocessing
from processSPP_util import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

 
# externalModelsNames = ['w2v_small','w2v_big']
externalModelsNames = ['w2v_uk1','w2v_big','w2v_ukfull','gloveWG100','gloveTW100']

    
def runSeries(nrTF=0,inline=0,summaryFile=None):
    #get correlations for both task types, both latencies, and both raw RT correlation + priming correlation
    #print results in latex tables
#     taskDirList = ['ldt','naming']
#     rawList = [1,0]
#     latencyList = ['200','1200']
    taskDirList = ['ldt']
    rawList = [1]
    latencyList = ['200']
    
    nrTF = bool(int(nrTF))
    
    for td in taskDirList:
        for r in rawList:
            for l in latencyList:
                print '%s-%s-%d'%(td,l,r)
                analyzeSPP(td,l,r,nr = nrTF,inline=inline,summaryFile=summaryFile)

def runSeriesLaTeX(nrTF=0,inline=0,summaryFile=None):
    #get correlations for both task types, both latencies, and both raw RT correlation + priming correlation
    #print results in latex tables
#     taskDirList = ['ldt','naming']
#     rawList = [1,0]
#     latencyList = ['200','1200']
    taskDirList = ['ldt']
    rawList = [1]
    latencyList = ['200']
    
    nrTF = bool(int(nrTF))
    
    for td in taskDirList:
        for r in rawList:
            for l in latencyList:
                print '%s-%s-%d'%(td,l,r)
                with open('texdocs/%s-%s-%d-%d-reltables.tex'%(td,l,r,int(nrTF)),'w') as out:
                    out.write(r'\documentclass{article}'+'\n')
                    out.write(r'\usepackage{multirow}'+'\n')
                    out.write(r'\usepackage[margin=.9in]{geometry}'+'\n')
                    out.write(r'\begin{document}'+'\n')
                    analyzeSPP(td,l,r,texTableDoc = out,nr = nrTF,inline=inline,summaryFile=summaryFile)
                    out.write(r'\end{document}')
                    

def analyzeSPP(task_dir,latency,raw,texTableDoc = None,nr = False,inline=0,summaryFile = None):

    if task_dir== 'ldt': t = 'LDT'
    elif task_dir== 'naming': t = 'NT'

    rawDep = '%s_%sms_Z'%(t,latency)
    pairDep = '%s_%sms_Z_Priming'%(t,latency) 
    
    if nr: 
        dist = 'NR'
        modValsPrefix = 'nrDF'
    else: 
        dist = 'SIM'
        modValsPrefix = 'simDF'
        
    modColsToAdd = ['%s_%s'%(n,dist) for n in externalModelsNames]
       
    if raw:
        modelList = ['LSA'] + modColsToAdd #names of columns to run correlations on
        dep = rawDep
    else:
        modelList = ['LSA_diff'] + modColsToAdd
        dep = pairDep
        
    tgtLexVars = []
    primeLexVars = [] 
    lexVars =  tgtLexVars + primeLexVars  
    
    relevantRegrVars = [dep] + modelList + lexVars
        
    if summaryFile:
        outFileName = '-'.join([dep,dist,'rels'])
        out = open(outFileName+'.txt','w')
    else: out = None
    
    
    #get dataframes
    targetLexDF,primeLexDF,rawDF,pairedDF,lexDict,relDict = readSPP(task_dir,t)
    
    #note this is a pointer to the original DF and you are changing that original DF when you change tempDF
    if raw: 
        tempDF = rawDF
        modValsDF = pandas.read_csv(modValsPrefix+'-raw.csv')
    else: 
        tempDF = pairedDF
        modValsDF = pandas.read_csv(modValsPrefix+'-paired.csv')

    
    #add lexical variables for targets and primes to the DF by looking up in dict
    tempDF = addLexVars(tempDF,lexDict,tgtLexVars,'_T','TargetWord')
    tempDF = addLexVars(tempDF,lexDict,primeLexVars,'_P','Prime') 
    if not raw: 
        addLexVars(tempDF,lexDict,primeLexVars,'_U','Unrelated')
        tempDF.insert(len(tempDF.columns.values),'LSA_diff',tempDF['LSA']-tempDF['LSA_U'])
    
    #insert VSM values into DF for all external models
#     for col in modColsToAdd:
#         tempDF.insert(len(tempDF.columns.values),NAME,modValsDF[col])
    if raw:
        for col in modColsToAdd:
            tempDF.insert(len(tempDF.columns.values),col,modValsDF[col])
    else:
        for col in modColsToAdd:
            if nr:
                unrel = modValsDF[col+'_un'].apply(math.log)
                rel = modValsDF[col+'_rel'].apply(math.log)
                tempDF.insert(len(tempDF.columns.values),col,unrel-rel)
            else:
                unrel = modValsDF[col+'_un']
                rel = modValsDF[col+'_rel']
                tempDF.insert(len(tempDF.columns.values),col,rel-unrel)
#     print tempDF
    
    colNamesTex = []
    rowNamesTex = []
    valsTex = []
    
    colNamesTex = [e for e in modelList]
    
    valTypes = ['rho: ','p: ','r: ','p: ','t: ','n: ']
    
    #get correlations for each model for full dataset
    row = []
    fullCoefs = []
    fullCoefErrs = []
    rowNamesTex.append('FullSPP')
    constraint = constrainDF('tempDF',relevantRegrVars)
    testDF = tempDF[eval(constraint)].reset_index()
    tot = len(testDF.iloc[:,0])
    for modelName in modelList:
        print '\n\nFull'
        print modelName
        if summaryFile:
            out.write('\nPREDICTOR MODEL: ' + modelName + '\n')
            out.write('\nALL ITEMS: %d\n'%tot)
        #regression
        coef,cil,tv,pv,rs,ra = regrSPP(testDF,dep,[modelName],1,out=out)
        
        #store regression coefficient and distance from coef to confidence interval lower bound 
        fullCoefs.append(coef)
        fullCoefErrs.append(coef-cil)
        
        #correlations
        (rho,rho_p),(r,p) = corrSPP(testDF,dep,modelName)
        
        prerow = [round(x,3) for x in [rho,rho_p,r,p,tv[0]]] + [tot]
        row.append([valTypes[i]+str(prerow[i]) for i in range(len(valTypes))])
        
        if summaryFile:
            out.write('Spearman rho: %f, p: %f\n'%(rho,rho_p))
            out.write('Pearson R: %f, p: %f\n\n'%(r,p))
    
    if texTableDoc:
        laTeXTable(texTableDoc,['FullSPP'],colNamesTex,[row])
    
    #plot full SPP coefficients
    plotPoints(fullCoefs,[m[0:7] for m in modelList],'fullSPP_coefs',errs=fullCoefErrs,inline=inline)
    
    valsTex.append(row)
    if summaryFile:
        out.write('-------------\n\n')
    
    
    #loop through relation types and get correlations within each relation, for each model
    #need to decide whether to compare models within a relation or relations within a model ... 
    #probably good to try both for now. see what is interesting
    rel2Mod = {}
    mod2Rel = {}
    relList = tempDF['Relation1'].unique()
    for rel in relList:
        constraint = constrainDF('tempDF',relevantRegrVars,['Relation1'],['==rel'],subsetDelim='&') 
        testDF = tempDF[eval(constraint)].reset_index()
        if rel not in testDF['Relation1'].unique(): continue
        tot = len(testDF.iloc[:,0])
        
        if tot < 10: continue
        rowNamesTex.append(rel[:6])
        row = []
#         try:
#             for var in relevantRegrVars:
#                 testDF[var] = sklearn.preprocessing.scale(map(float,testDF[var]))
#         except: print "\nWARNNG: COULDN'T SCALE\n"
        rel2Mod[rel] = {}
        for modelName in modelList:
            print '\n\n'+ rel
            print modelName
            if not modelName in mod2Rel: mod2Rel[modelName] = {}
            if summaryFile:
                out.write('\nPREDICTOR MODEL: ' + modelName + '\n')
                out.write('RELATION: ' + str(rel) + '\n')
                out.write('TOTAL ITEMS: ' + str(tot) + '\n')
            #regression
            coef,cil,tv,pv,rs,ra = regrSPP(testDF,dep,[modelName],1,out=out)
            
            #store regression coefficient and distance from coef to confidence interval lower bound in re2Mod and mod2Rel dicts so we can plot in both directions later
            rel2Mod[rel][modelName] = (coef,coef-cil)
            if not rel in mod2Rel[modelName]: mod2Rel[modelName][rel] = {}
            mod2Rel[modelName][rel] = (coef,coef-cil)
            
            #correlations
            (rho,rho_p),(r,p) = corrSPP(testDF,dep,modelName)
            
            prerow = [round(x,3) for x in [rho,rho_p,r,p,tv[0]]] + [tot]
            row.append([valTypes[i]+str(prerow[i]) for i in range(len(valTypes))])
            
            if summaryFile:
                out.write('Spearman rho: %f, p: %f\n'%(rho,rho_p))
                out.write('Pearson R: %f, p: %f\n\n'%(r,p))
        
        if texTableDoc:
            laTeXTable(texTableDoc,[rel],colNamesTex,[row])
            
        if summaryFile:
            out.write('-------------\n\n')
    
#     if texTableDoc:
#         laTeXTable(texTableDoc,rowNamesTex,colNamesTex,valsTex)

    #iterate through relations and plot across models for that relation
    for rel in rel2Mod:
        #iterate through models for this given relation and make a list of them and plot
        #new plot for each relation
        relCoefs = []
        relCoefErrs = []
        mList = []
        for mod in rel2Mod[rel]:
            mList.append(mod)
            relCoefs.append(rel2Mod[rel][mod][0])
            relCoefErrs.append(rel2Mod[rel][mod][1])
        plotPoints(relCoefs,[m[0:7] for m in mList],'%s_coefs'%rel,errs=relCoefErrs,inline=inline)
    #iterate through models and plot across relations for that model
    for mod in mod2Rel:
        modCoefs = []
        modCoefErrs = []
        rList = []
        for rel in mod2Rel[mod]:
            rList.append(rel)
            modCoefs.append(mod2Rel[mod][rel][0])
            modCoefErrs.append(mod2Rel[mod][rel][1])
        plotPoints(modCoefs,[r[0:3] for r in rList],'%s_coefs'%mod,errs=modCoefErrs,inline=inline)
    if summaryFile:
        out.close()


def readSPP(task_dir,t):

    lexDict = {}
    relDict = {}
    lexItemsToFix = [('CELCIUS','CELSIUS'),('SKILLER','SKILLET'),('CONDFIDENCE','CONFIDENCE')]
    relItemsToFix = [('.*unclas','unclassified'),('.*anton','antonym'),('Instrument','instrument')]
    rowsToRemove = ['yes','stairs','joke'] #assocrel and assocunrel pairs need to be removed because the unrelated rows are repeats
    
    lexList = ['Length','SubFreq','LogSubFreq','LogHAL','ELP_'+t+'_RT','ELP_'+t+'_Z','POS']
    relList2 = ['Relation1','LSA','FAS','BAS',t+'_200ms_Z_Priming',t+'_1200ms_Z_Priming',t+'_200ms_RT_Priming',t+'_1200ms_RT_Priming']
    relList = ['Relation1','LSA','FAS','BAS',t+'_200ms_RT',t+'_1200ms_RT',t+'_200ms_Z',t+'_1200ms_Z','Related','Assoc']
    
    for file in os.listdir(os.path.abspath(task_dir)):
        if re.match('assoc_.*\.csv',file): assocRel = os.path.join(task_dir,file)
        elif re.match('assocunrel_.*\.csv',file): assocUn = os.path.join(task_dir,file)
        elif re.match('otherassoc_.*\.csv',file): otherAssocRel = os.path.join(task_dir,file)
        elif re.match('otherassocunrel_.*\.csv',file): otherAssocUn = os.path.join(task_dir,file)
        elif re.match('target_.*\.csv',file): targetF = os.path.join(task_dir,file)
    
    frameTGT = pandas.read_csv(targetF)
    l = len(frameTGT['TargetWord'])
    
    frameAR = pandas.read_csv(assocRel)
    frameOAR = pandas.read_csv(otherAssocRel)

    frameAU = pandas.read_csv(assocUn)
    frameOAU = pandas.read_csv(otherAssocUn)
    
    excludeRows = (frameAR['TargetWord'] != 'yes')&(frameAR['TargetWord'] !='stairs')&(frameAR['TargetWord'] !='joke')&(frameAR['TargetWord'] !='guitar')&(frameAR['TargetWord'] !='language')
    frameAR = frameAR[excludeRows].reset_index()
    frameAU = frameAU[excludeRows].reset_index()
    
    excludeRows2 = (frameOAR['TargetWord'] !='pilot')&(frameOAR['TargetWord'] !='train')
    frameOAR = frameOAR[excludeRows2].reset_index()
    frameOAU = frameOAU[excludeRows2].reset_index()
    
    for df in [frameAR,frameOAR]: df['Related'] = 'REL'
    for df in [frameAU,frameOAU]: 
        df['Related'] = 'UNREL'
        df['Relation1'] = 'norel'
        df['FAS'] = None
        df['BAS'] = None
    for df in [frameAR,frameAU]: df['Assoc'] = 'ASSOC'
    for df in [frameOAR,frameOAU]: df['Assoc'] = 'OTHER'

    
    [frameTGT,frameAR,frameOAR,frameAU,frameOAU] = colSub([frameTGT,frameAR,frameOAR,frameAU,frameOAU])
    [frameTGT,frameAR,frameOAR,frameAU,frameOAU] = lowerConvert([frameTGT,frameAR,frameOAR,frameAU,frameOAU],['TargetWord','Prime'])
    
    for i in range(len(frameOAR.iloc[:,1])):
        tgtWord = frameOAR.ix[i,'TargetWord']
        prime = frameOAR.ix[i,'Prime']
        relDict[tgtWord] = {}
        relDict[tgtWord]['RelAO'] = prime
    for i in range(len(frameOAU.iloc[:,1])):
        tgtWord = frameOAU.ix[i,'TargetWord']
        prime = frameOAU.ix[i,'Prime']
        relDict[tgtWord]['RelUO'] = prime
    for i in range(len(frameAR.iloc[:,1])):
        tgtWord = frameAR.ix[i,'TargetWord']
        prime = frameAR.ix[i,'Prime']
        if tgtWord not in relDict: 
            relDict[tgtWord] = {}
        relDict[tgtWord]['RelA'] = prime
    for i in range(len(frameAU.iloc[:,1])):
        tgtWord = frameAU.ix[i,'TargetWord']
        prime = frameAU.ix[i,'Prime']
        relDict[tgtWord]['RelU'] = prime
    
    mismatch = [e for e in frameAR['Prime'] if e not in list(frameAU['Prime'])]
#     mismatch = [e for e in frameAU['Prime'] if e not in list(frameAR['Prime'])]

    targetLexDF = frameTGT[['TargetWord']+lexList]
    primeLexDF = pandas.merge(frameAR[['Prime']+lexList],frameOAR[['Prime']+lexList], how='outer')
    primeLexDF = itemSub(primeLexDF,lexItemsToFix,'Prime')
    
    for i in range(len(targetLexDF['TargetWord'])):
        wrd = targetLexDF.ix[i,'TargetWord'].lower()
        lexDict[wrd] = {}
        for it in lexList: lexDict[wrd][it] = targetLexDF.ix[i,it]
    for i in range(len(primeLexDF['Prime'])):
        wrd = primeLexDF.ix[i,'Prime'].lower()
        lexDict[wrd] = {}
        for it in lexList: lexDict[wrd][it] = primeLexDF.ix[i,it]
                
    relPreDF1 = pandas.merge(frameAR[['TargetWord','Prime']+relList],frameOAR[['TargetWord','Prime']+relList], how='outer')
    relPreDF2 = pandas.merge(frameAU[['TargetWord','Prime']+relList],frameOAU[['TargetWord','Prime']+relList], how='outer')
#     
    relDF = pandas.merge(relPreDF1,relPreDF2,how='outer')
    relDF = itemSub(relDF,lexItemsToFix,'Prime')
    relDF = itemSub(relDF,relItemsToFix,'Relation1')
    
    #add unrelated primes column to assocrel
    pairedPreDF1 = frameAR[['TargetWord','Prime'] + relList2]
    pairedPreDF1.insert(2,'Unrelated',frameAU['Prime'])
    pairedPreDF1.insert(len(pairedPreDF1.columns.values),'LSA_U',frameAU['LSA'])
    pairedPreDF1.insert(len(pairedPreDF1.columns.values),'NormStat','NORM')
    
    pairedPreDF2 = frameOAR[['TargetWord','Prime'] + relList2]
    pairedPreDF2.insert(2,'Unrelated',frameOAU['Prime'])
    pairedPreDF2.insert(len(pairedPreDF2.columns.values),'LSA_U',frameOAU['LSA'])
    pairedPreDF2.insert(len(pairedPreDF2.columns.values),'NormStat','OTHER')
    
    
    pairedDF = pandas.merge(pairedPreDF1,pairedPreDF2,how='outer')
    pairedDF = itemSub(pairedDF,lexItemsToFix,'Unrelated')
    pairedDF = itemSub(pairedDF,relItemsToFix,'Relation1')

    for i in range(len(pairedDF.iloc[:,1])):
        tg = pairedDF.ix[i,'TargetWord']
        p = pairedDF.ix[i,'Prime']
        u = pairedDF.ix[i,'Unrelated']
        r = pairedDF.ix[i,'NormStat']
        if r == 'NORM':
            if relDict[tg]['RelA'] != p or relDict[tg]['RelU'] != u: print ' '.join(tg,p,u)
        elif r == 'OTHER':
            if relDict[tg]['RelAO'] != p or relDict[tg]['RelUO'] != u: print ' '.join(tg,p,u)
#         z200p = pairedDF.ix[i,t+'_200ms_Z_Priming']
#         z200rel = relDF[t+'_200ms_Z'][(relDF['TargetWord'] == tg)&(relDF['Prime'] == p)].values[0]
#         z200unrel = relDF[t+'_200ms_Z'][(relDF['TargetWord'] == tg)&(relDF['Prime'] == u)].values[0]
#         primingComp = z200unrel - z200rel
#         if abs(z200p - primingComp) > .000001:
#             print tg
#             print p
#             print u
#             print abs(z200p - primingComp)
#             print '\n'

    
    return targetLexDF,primeLexDF,relDF,pairedDF,lexDict,relDict
               
if __name__ == "__main__":
    runSeries()
#     synSetEval(synFileNames,externalModels,externalModelsNames)
#     with open('table.tex','w') as out:
#         laTeXTable(out,[1,2],[1,2,3,4],mat)
#     constrainDF(targetLexDF,['SubFreq'],['Length','Length'],['==5','==6'],subsetDelim='|')
