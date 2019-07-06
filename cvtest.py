from STAT import Data,Analyzer,Simu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
test random sample propertys
"""
def samplesize_CV(n):
    cv=n
    mean=100
    result={i:[] for i in range(2,21)}
    fin=dict.fromkeys(result)
    for s in result.keys():
        for i in range(100000):
            _=Analyzer.cv_cacu(1, np.random.normal(mean,cv*mean/100,s))
            result[s].append(_)
    for k,i in result.items():
        fin[k]=np.percentile(i,q=[2.5,97.5])
    return fin
        # print(k,np.percentile(i,q=[2.5,97.5]))

    # fig,ax = plt.subplots(figsize=(8,6))
    # for i,j in result.items():
    #     _=ax.hist(j,bins=700,histtype='step',density=True,label=str(i))
    # ax.legend()
    # ax.set_title("CV distribution - sample size, CV={}, Mean={}".format(cv,mean))
    # ax.set_xlabel("CV")
    # ax.set_ylabel('Frequency')
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('CV={}-sample size.svg'.format(n))



result=[]
for i in range(1,21,1):
    result.append(samplesize_CV(i))

def series(x):
    return x.apply(lambda x:"{:>6.2f} -{:>6.2f}".format(x[0],x[1]))


df=pd.DataFrame(result)

df=df.apply(series)

df.index=["CV={}".format(i) for i in range(1,21,1)]
df.columns=["Size={}".format(i) for i in range(2,21)]
df.to_csv('cv_size.csv')
