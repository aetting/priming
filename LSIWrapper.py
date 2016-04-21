import gensim, os, sys, gzip, re


class MyDocs(object):
    def __init__(self,dirlist):
        self.dirlist = dirlist

    def __iter__(self):
        for d in self.dirlist:
            for f in os.listdir(d):
                doc = []
                if f.endswith('.gz'): fileObject = gzip.open(os.path.join(d,f))
                else: fileObject = open(os.path.join(d,f))
                for line in fileObject:
                    if re.match('<\/doc',line): 
                        yield doc
                        doc = []
                        continue
                    doc.append(line)

dirlist = [sys.argv[i] for i in range(2,len(sys.argv))]
saveto = sys.argv[1]
corp = MyDocs(dirlist)
# for x in corp:
#     print x
#     print 'X\nX\nX\nX\nX\n'
model = gensim.models.lsimodel.LsiModel(corpus=corp,num_topics=100,chunksize=50)
model.save(saveto)
