from .simulator import Simu
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations,product
import pandas as pd
from functools import partial



class StatTools(Simu):
    def plot_hist(self,*data,**kwargs):
        bins=kwargs.get('bins',700)
        histtype=kwargs.get('histtype','step')
        density=kwargs.get('density',True)
        for item in data:
            fig,ax = plt.subplots(figsize=(8,6))
            if isinstance(item,dict):
                for i,j in item.items():
                    _=ax.hist(j,bins=bins,histtype=histtype,density=density,label=str(i))
            else:
                _=ax.hist(item,bins=bins,histtype=histtype,density=density,)
            ax.legend()
            ax.set_title(kwargs.get('title','title'))
            ax.set_xlabel(kwargs.get("xlabel",'xlabel'))
            ax.set_ylabel(kwargs.get("ylabel","ylabel"))
            plt.tight_layout()
            if kwargs.get('save',False):
                save=self.ifexist((kwargs.get('save'))+'.svg')
                plt.savefig(save)
            else:plt.show()

    def plot_cv_n(self,cv,size,n=1e6,q=95,mean=100,plot=True,save=False,**kwargs):
        """
        plot the density curve given cv and ssample size.
        also return dict of CV ramdomly generated.
        """
        cvs=[cv] if isinstance(cv,(int,float)) else cv
        sizes=[size] if isinstance(size,(int,float)) else size
        result={}
        for cv,size in product(cvs,sizes):
            rand = np.random.normal(mean,cv*mean/100,size*int(n)).reshape((size,-1))
            calccv = np.std(rand,ddof=1,axis=0)/np.mean(rand,axis=0)*100
            ci=np.percentile(calccv,q=[50-q/2,50+q/2])
            result['CV{:<2}SZ{:<2}; {:.1f}-{:.1f}'.format(cv,size,*ci)]=calccv
        if plot:
            a=Simu()
            a._plot_histgram(result,save=save,stat="CV",title='CV-Size plot',subtitle='CV-Size',log=False,**kwargs)
        return result

    def _mc_CV_p(self,cv1,cv2,s1,s2,n=1e6,**kwargs):
        """
        calculate how likely two sample will be this apart if they have the same cv
        """
        n=int(n)
        avgcv=(cv1*s1+cv2*s2)/(s1+s2)
        sample1 = np.random.normal(100,avgcv,n*s1).reshape(s1,-1)
        sample2 = np.random.normal(100,avgcv,n*s2).reshape(s2,-1)
        sampl1cv = np.std(sample1,ddof=1,axis=0)/np.mean(sample1,axis=0)*100
        sampl2cv = np.std(sample2,ddof=1,axis=0)/np.mean(sample2,axis=0)*100
        delta=np.sum(np.abs(sampl1cv-sampl2cv) >= np.abs(cv1-cv2))
        return delta/n


    def _mc_Mean_p(self,m1,m2,std1,std2,s1,s2,n=int(1e6),**kwargs):
        avg=(m1*s1+m2*s2)/(s1+s2)
        sample1 = np.random.normal(avg,std1,n*s1).reshape(s1,-1)
        sample2 = np.random.normal(avg,std2,n*s2).reshape(s2,-1)
        sampl1mean = np.mean(sample1,axis=0)
        sampl2mean = np.mean(sample2,axis=0)
        delta=np.sum(np.abs(sampl1mean-sampl2mean) >= np.abs(m1-m2))
        return delta/n

    def MC_analysis(self,*args,n=1e6,**kwargs):
        """
        monte carlo analysis of mean or cv p value.
        input: multiple data set or named input of data. (list or arrays)
        """
        methoddict={"cv":self._mc_CV_p,"mean":self._mc_Mean_p}
        method=kwargs.pop('method','mean')
        try:func = methoddict[method]
        except: raise ValueError('unsupported method')
        if all([isinstance(i,(int,float)) for i in args]):
            if len(args)==4:
                func=methoddict['cv']
            return func(*args)
        save=kwargs.pop('save',False)
        for i,j in enumerate(args):
            kwargs.update({'input'+str(i+1):j})
        df=pd.DataFrame(np.zeros((len(kwargs),len(kwargs)),dtype=float),index=kwargs.keys(),columns=kwargs.keys())
        for (k1,i1),(k2,i2) in combinations((kwargs.items()),2):
            std1,std2=np.std(i1,ddof=1),np.std(i2,ddof=1)
            m1,m2 = np.mean(i1),np.mean(i2)
            cv1,cv2 =std1/m1,std2/m2
            s1,s2=len(i1),len(i2)
            para=dict(std1=std1,std2=std2,m1=m1,m2=m2,cv1=cv1,cv2=cv2,s1=s1,s2=s2)
            df.loc[k1,k2]=df.loc[k2,k1]=func(**para,n=int(n))
        if save:
            save=self.ifexist('MC_{}_analysis.xlsx'.format(method))
            self._write_df_toxlsx(save,df)
        else:
            return df




def find_sep(arg,f=None,interval=None):
    init=arg[0]
    cv2=arg[0]
    s=arg[1]
    func=partial(f,cv2=cv2,s1=s,s2=s,n=2e5)
    sep=func(init)
    up=[init]
    low=[]
    count=0
    while sep < interval-0.001 or sep>interval+0.001:
        count+=1
        if not low:
            init+=2
        else:
            init = 0.5*(up[-1]+low[-1])
        sep=func(init)
        if sep>interval+0.001:
            up.append(init)
        elif sep < interval-0.001:
            low.append(init)
        sep=func(init)
    return init
