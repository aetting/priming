#!/usr/bin/python

import os, re, csv, scipy, sklearn, gensim, pandas, math, gzip
from scipy import stats,spatial
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from matplotlib import pyplot as plt

def getNR(testw,wordstorank,mod):
    words = []
    sims = []
    ranksToReturn = []
    if type(mod) == dict: voc = mod
    else: voc = mod.vocab
    for w in voc:
        s = cosSim(w,testw,mod)
        words.append(w)
        sims.append(s)
    b = np.argsort(sims)[::-1]
#     print [words[b[i]] for i in range(10)]
    #dict mapping words to rankings
    d = {words[b[i]]:(i) for i in range(len(b))}
    for wtr in wordstorank:
        try: d[wtr]
        except: 
            print 'FAIL'
            r = np.nan
        else: r = d[wtr]
        ranksToReturn.append(r)  
    return ranksToReturn
            
            
def readVectors(filename):
    if filename.endswith('.gz'):
        fileObject = gzip.open(filename, 'r')
    else:
        fileObject = open(filename, 'r')
    
    wordVectors = {}
    for line in fileObject:
        s = line.lower().strip().split()
        word = s[0]
        vector = np.array(map(float,s[1:]))
        wordVectors[word] = vector
        
    fileObject.close()
        
    return wordVectors
    
def cosSim(u,v,model):
    try: 
        u = model[u]
        v = model[v]
    except: 
        sim = np.nan
    else: sim = (1 - scipy.spatial.distance.cosine(u,v))
    return sim
    
def laTeXTable(out,rownames,colnames,mat):
    colnames = [re.sub('_','',e) for e in colnames]
    rownames = [re.sub('_','-',e) for e in rownames]
    out.write(r'\begin{tabular}{')
    for i in range(len(colnames)+1):
        out.write('c')
    out.write(r'|}\cline{2-%s}'%(len(colnames)+1)+'\n')
    if len(colnames) == 1: firstcell = '|c|'
    else: firstcell = '|c'
    out.write(r'&\multicolumn{1}{%s}{%s}%s'%(firstcell,colnames[0],''.join([' & '+str(e) for e in colnames[1:]])) + r' \\\hline'+ '\n')
    for i in range(len(rownames)):
        try: len(mat[i][0])
        except: lenmat = 1
        else: lenmat = len(mat[i][0])
        if lenmat == 1:
            out.write(r'\multicolumn{1}{|c|}{\multirow{%d}{*}{%s}} & %s'%(lenmat,rownames[i],' & '.join([str(c) for c in mat[i]]))+ r' \\'+ '\n' )
        else:
            out.write(r'\multicolumn{1}{|c|}{\multirow{%d}{*}{%s}} & %s'%(lenmat,rownames[i],' & '.join([str(c[0]) for c in mat[i]]))+ r' \\'+ '\n' )
            for j in range(1,lenmat):
                out.write(r'\multicolumn{1}{|c|}{} & %s'%(' & '.join([str(c[j]) for c in mat[i]]))+ r' \\'+ '\n' )
        out.write(r'\hline'+'\n')
    out.write(r'\end{tabular}\\'+'\n')
    
def regrSPP(df,depcol,predlist,scale,out=None):
    if scale:
        try:
            for var in predlist + [depcol]:
                df[var] = sklearn.preprocessing.scale(map(float,df[var]))
        except: print "\nWARNNG: COULDN'T SCALE\n"
    y = df[depcol]
#     print y
#     print 'Y stats'
#     print np.average(y)
#     print np.std(y)
    X = df[predlist]
#     for p in predlist:
 #        print df[p]
#         print 'PREDICTOR stats'
#         print np.average(df[p])
#         print np.std(df[p])
    mod = sm.OLS(y,X)
    results = mod.fit()
#     print results.summary()
    
#     get regression coefficient and lower and upper bounds of confidence interval
    coef = results.params[0]
#     print coef
    cil = results.conf_int()[0][0]

#     print results.summary()
    toReturn = [coef,cil,results.tvalues,results.pvalues,results.rsquared,results.rsquared_adj]
    if out:
        for i in range(len(results.tvalues)):
            out.write('t: %f,p: %f\n'%(results.tvalues[i],results.pvalues[i]))
        out.write('R-squared: %f\n'%results.rsquared)
        out.write('R-squared (adj): %f\n\n'%results.rsquared_adj)
    
    return toReturn
      

def corrSPP(df,col1,col2):
#     targetLexDF,primeLexDF,relDF,lexDict = readSPP(task_dir)
    constraint = (df[col1].notnull()) & (df[col2].notnull())
    v1 = df[col1][constraint]
    v2 = df[col2][constraint]
    rho,p_rho = scipy.stats.spearmanr(v1,v2)
    r,p = scipy.stats.pearsonr(v1,v2)
    return (rho,p_rho),(r,p)
    
    
    
def addSimCols(df,models,modelNames,w1colName,w2colName):
    for m in range(len(models)):
        model = models[m]
        modelName = modelNames[m]
        df.insert(len(df.columns.values),modelName,np.nan)
        for i in range(len(df.iloc[:,1])):
            w1 = df.ix[i,w1colName]
            w2 = df.ix[i,w2colName]
            try: 
                model[w1]
                model[w2]
            except: 
                continue
            else: 
                sim = cosSim(w1,w2,model)
            df.set_value(i,modelName,sim)
    return df

            
def constrainDF(dfName,relevantColsList,subsetOnColList=None,subsetValList=None,subsetDelim='&'):
    #subsetOnCol = column to subset on
    #subsetVal = description of right half of logical statement describing how to subset (e.g. '==5')
    #take df and remove any row with empty values in any relevant column (probably all predictors being combined OR compared)
    #also reduce to only desired rows based on some criterion
    if subsetOnColList:
        assert (len(subsetOnColList) == len(subsetValList)),"different number of subsetting columns and statements!"
    notNullList = []
    subsetList = []
    for c in relevantColsList:
        s = "(%s['%s'].notnull())"%(dfName,c)
        notNullList.append(s)
    if subsetOnColList:
        for i in range(len(subsetOnColList)):
            subsetOnCol = subsetOnColList[i]
            subsetVal = subsetValList[i]
            subsetList.append("(%s['%s']%s)"%(dfName,subsetOnCol,subsetVal))
        constraint = '&'.join(notNullList)+'&('+subsetDelim.join(subsetList)+')'
    else: constraint = '&'.join(notNullList)

    return constraint
    
    
def addLexVars(dfToUpdate,sourceDict,lexVarNames,suffix,wordColName):
    #insert lexical vars into DF
    #wordColName is the column to look in for the word to look up for vars in the dict
    for v in lexVarNames:
        name = v+suffix
        dfToUpdate.insert(len(dfToUpdate.columns.values),name,np.nan)
        for i in range(len(dfToUpdate.iloc[:,1])):
            word = dfToUpdate.ix[i,wordColName].lower()
            try: dfToUpdate.set_value(i,name,sourceDict[word][v])
            except: 
                dfToUpdate.set_value(i,name,np.nan)
            
    return dfToUpdate

def colSub(dfs):
    newdfs = []
    for df in dfs:
        if 'Unrelated' in df.columns.values: df = df.rename(columns={'Unrelated':'Prime'})
        for colName in df.columns.values:
            if re.match('.*\s',colName): df = df.rename(columns={colName:re.sub(' ','_',colName)})
        newdfs.append(df)
    return newdfs

def itemSub(df,fixList,column): 
    for i in range(len(df.iloc[:,1])):
        for fix in fixList:
            pattern = fix[0]
            correction = fix[1]
            if re.match(pattern,str(df.ix[i,column])):
                df.set_value(i,column,correction)
    return df
    
def lowerConvert(dfs,colsToLower):
    for df in dfs:
        for col in colsToLower:
            if col not in df.columns.values: continue
            df[col] = df[col].str.lower()
    return dfs
    
def synSetEval(synFileNames,models,modelNames,texTableDoc = None):
    lurelist = ['Lure1','Lure2','Lure3']
    rownames = []
    tableVals = []
    for synFileName in synFileNames:
        row = []
        setName = re.match('.+/([^/]+)$',synFileName).group(1)
        rownames.append(setName)
        df = pandas.read_csv(synFileName,delimiter = '\s*\|\s*',engine='python',header=None,names=['Probe','Correct','Lure1','Lure2','Lure3'])
        for m in range(len(models)):
            model = models[m]
            modelName = modelNames[m]
            df.insert(len(df.columns.values),modelName,np.nan)
            for i in range(len(df.iloc[:,1])):
                simList = []
                probe = df.ix[i,'Probe']
                correct = df.ix[i,'Correct']
                lures = [df.ix[i,e] for e in lurelist]
                check = [correct] + lures
                maxSim = 0
                try: model[probe]
                except:
                    winner = None
                    continue
                for w in check:
                    try: model[w]
                    except: 
                        winner = None
                        break
                    else:
                        sim = cosSim(probe,w,model)
                        if sim > maxSim: 
                            maxSim = sim
                            winner = w
                if not winner: continue
                if winner == correct: 
                    df.set_value(i,modelName,1)
                else: 
                    df.set_value(i,modelName,0)
        #level of model loop
        constraint = constrainDF('df',modelNames)
        testDF = df[eval(constraint)]
        tot = len(testDF.iloc[:,0])
        for name in modelNames:
            acc = sum(testDF[name])/float(tot)
            row.append([round(100*acc,2),tot])
        tableVals.append(row)
    if texTableDoc:
        laTeXTable(texTableDoc,rownames,modelNames,tableVals)
        
                
def simSetEval(simFileNames,models,modelNames,texTableDoc = None):
    rownames = []
    tableVals = []
    valTypes = ['rho: ','p: ','r: ','p: ','n: ']
    for simFileName in simFileNames:
        simDF = pandas.read_csv(simFileName,delimiter = '\s',engine='python',header=None,names=['w1','w2','SimRating'])
        simDF['w1'] = simDF['w1'].apply(lambda x: re.sub('-.*','',x))
        simDF['w2'] = simDF['w2'].apply(lambda x: re.sub('-.*','',x))
        simDF = addSimCols(simDF,models,modelNames,'w1','w2')
    
        constraint = constrainDF('simDF',modelNames)
        testDF = simDF[eval(constraint)]
        tot = len(testDF.iloc[:,0])
    
        row = []
        setName = re.match('.+/([^/]+)$',simFileName).group(1)
        rownames.append(setName)
        for m in range(len(models)):
            modelName = modelNames[m]
            (rho,rho_p),(r,p) = corrSPP(testDF,'SimRating',modelName)
            prerow = [round(x,3) for x in [r,p,rho,rho_p]] + [tot]
            row.append([valTypes[i]+str(prerow[i]) for i in range(len(valTypes))])
        tableVals.append(row)
    if texTableDoc:
        laTeXTable(texTableDoc,rownames,modelNames,tableVals)
        
def plotPoints(pts,xlabels,name,errs=None,inline=0):
#     markers = ['x','v','s']
    for i in range(len(pts)):
        p = pts[i]
#         plt.axhline(p,.35,.65,c='r')
#         plt.scatter(i,p,marker=markers[i],label=labels[i],c='black',s=50)
        plt.scatter(i,p,c='black',s=50)
        if errs:
            plt.errorbar(i,p,yerr=errs[i],c='black')
#         plt.hlines(pts,.8,1.2,colors='r')
#     plt.legend(scatterpoints=1)
#     plt.xlabel('Context type')
#     plt.xlim(0,3)
    plt.xticks(range(len(pts)),xlabels)
#     plt.ylabel('N400 peak amplitude',fontsize='x-large')
    plt.tick_params(axis='x', which='both', bottom='off', top='off',labelsize='small')
    plt.tick_params(axis='y', which='both',labelsize='small')
#     plt.gca().invert_yaxis()
    if inline:
        plt.title(name)
        plt.show()
    else:
        plt.savefig('plots/%s.png'%name)
    plt.clf()
    
def plotAxis(ax,pts,xlabels,name,errs=None,inline=0):
    for i in range(len(pts)):
        p = pts[i]
        print p
        ax.scatter(i,p,c='black',s=50)
        if errs:
            ax.errorbar(i,p,yerr=errs[i],c='black')
    #figure out how to set individual labels
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels(xlabels,size='x-small')
    ax.set_title(name)
#     ax.tick_params(axis='x', which='both', bottom='off', top='off',labelsize='small')
#     ax.tick_params(axis='y', which='both',labelsize='small')
#     if inline:
#         plt.title(name)
#         plt.show()
#     else:
#         plt.savefig('plots/%s.png'%name)
#     plt.clf()
    
def plotTogether():
#     x = np.linspace(0, 2 * np.pi, 400)
#     y = np.sin(x ** 2)
    x = np.array([1,2,3])
    y = np.array([1,4,8])
    f, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(3, 2, sharex='col', sharey=True)
    axList =[ax1,ax2]
    for i in range(len(axList)):
        plotAxis(axList[i],[1,4,8],['w','e','r'],'test',errs=[7,4,5])
#     ax1.plot(x, y)
#     ax1.set_title('Sharing x per column, y per row')
#     ax2.scatter(x, y)
    ax3.scatter(x, 2 * y , color='r')
    ax4.plot(x, 2 * y, color='r')
    ax5.scatter(x, 2 * y , color='g')
    ax6.plot(x, 2 * y, color='g')
    plt.show()

    
if __name__ == "__main__":
    plotTogether()

#     rank = getNR('cat','elephant',model)
#     print rank