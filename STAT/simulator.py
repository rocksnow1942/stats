import numpy as np
from .cv_analysis import Analyzer
import pandas as pd
from os import path,getcwd

class Simu(Analyzer):
    def __init__(self,file=None,conc=[0,0.25,0.5,0.75,1],sig=[100,125,150,175,200],
                cv=[5,5,5,5,5],size=5):
        """
        default y = 90x+100; CV=7% on all points.
        """
        if file:
            data=self.readfile(file)
            self.__init__(size=size,**data)
            filefolder = path.dirname(file)
            savefolder= path.join(filefolder,'simu_analysis')
            self.save_loc=savefolder
            self.seed_df=pd.DataFrame(data,index=data.pop('conc'))
        else:
            data = dict.fromkeys(conc)
            size = size if isinstance(size,list) else [size]*len(conc)
            assert len(conc)==len(cv)==len(sig)==len(size), ("lenth not uniform")
            for c,s,cv_,si in zip(conc,sig, cv, size):
                data[c]=list(np.random.normal(s,cv_*s/100.0,si))
            Analyzer.__init__(self,data)
            self.seed=dict(conc=conc,sig=sig,cv=cv,size=size)
            self.save_loc= getattr(self,"save_loc",getcwd())

    def readfile(self,filename):
        df = pd.read_excel(filename)
        conc=df['Conc.'].tolist()
        sig=df['Raw_Mean'].tolist()
        cv=df['Raw_CV'].tolist()
        return dict(conc=conc,sig=sig,cv=cv)

    def set_conc(self,conc,point=None):
        if point:
            self.seed['conc']=list(np.linspace(*conc,point))
        else:
            self.seed['conc']=conc
        n=len(self.seed['conc'])
        self.seed['sig']=list(np.linspace(self.seed['sig'][0],self.seed['sig'][-1],n))
        self.seed['cv']=[np.mean(self.seed['cv'])]*n
        self.seed['size']=self.seed['size'][0:1]*n
        self.__init__(**self.seed)

    def set_CV(self,cv):
        self.seed['cv']= cv if isinstance(cv,list) else [cv]*len(self.seed['cv'])
        self.__init__(**self.seed)

    def set_size(self,size):
        self.seed['size']=size if isinstance(size,list) else [size]*len(self.seed['size'])
        self.__init__(**self.seed)

    def set_signal_bkgd(self,sig,bkgd):
        conc=self.seed['conc']
        cmin,cmax=min(conc),max(conc)
        self.seed['sig']=[bkgd + sig*(i-cmin)/(cmax-cmin) for i in conc]
        self.__init__(**self.seed)

    def generate_reading(self,conc=None):
        result = []
        for c,s,cv in zip(self.seed['conc'],self.seed['sig'],self.seed['cv']):
            result.append(np.random.normal(s,cv*s/100.0))
        return result

    def simu_accuracy(self,n=100,save=False,refreshfitting=True,**kwargs):
        """
        should simulate accuracy using the confusion matrix
        """
        # result = {i:[] for i in self.seed['conc']}
        temp=np.zeros((n,len(self.seed['conc'])),dtype=float)
        for i in range(n):
            measurement=self.generate_reading()
            if refreshfitting:
                self.__init__(**self.seed)
                _=self.fit()
            temp[i,:]=self.read_fib_conc(measurement)
        result={i:temp[:,k] for k,i in enumerate(self.seed['conc'])}
        save = save and "SimuAcc"
        stats="CV{:.2g}%_SN{:.2g}_Sz{:.2g}".format(np.mean(self.seed['cv']),
        max(self.seed['sig'])/min(self.seed['sig']),np.mean(self.seed['size']))
        self._plot_histgram(title="SimuAcc",save=save,result=result,
                stat=stats,repeats=n,**kwargs)
        # return confusion_matrixx
