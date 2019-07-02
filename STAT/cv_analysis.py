import pandas as pd
import STAT.fit_binding as fb
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.stats import t
from functools import partial
from os import path,makedirs

global savefolder,plotbackend

plotbackend = '.svg'


def workfunc(data,kw,st,xT,yT):
    re=Analyzer(data,xT=xT,yT=yT,transform=False)
    result = dict.fromkeys(st)
    if any([("residuals" in stat) or ("predict" in stat) for stat in st]):
        r = getattr(re,"residuals")
        residual = r(**kw)
        kw.update(resi=residual)
    for stat in st:
        r = getattr(re,stat)
        result[stat]=r(**kw)
    return result



class DataMixin:
    def _convert(self,data,method):
        """
        holds different methods to treat data.
        'avg': average on non zero values.
        'T'/'M'/'B': select top or middle or bottom electrode on all chips.
        int 1-5: select i-th data point on one electrode.
        """
        if method=='avg':
            return np.mean([i for i in data if i])
        elif str(method) in 'TMB':
            return data['TMB'.index(method)]
        elif isinstance(method,int):
            return data[method-1]
        elif method=='sd_rmo':
            gap=np.std(data,ddof=1) * 3
            mid=np.mean(data)
            return np.mean([i for i in data if (mid-gap)<=i<=(mid+gap)])
        elif method=='p_rmo':
            q1,q3=np.percentile(data,q=[25,75])
            lb = q1-1.5*(q3-q1)
            ub = q3+1.5*(q3-q1)
            return np.mean([i for i in data if lb<=i<=ub])
        elif method=='mid':
            data = sorted([i for i in data if i])
            return np.mean(data[1:-1])
        else:
            try:
                return data[int(method)-1]
            except:
                pass


    def rmoutlier(self,data,method='stdev-3'):
        """
        or use Q-1.5 for quantiles; default 1.5
        """
        method = method if isinstance(method,str) else 'stdev-3'
        if method.startswith('Q'):
            if '-' not in method: method+='-1.5'
            r = float(method.split('-')[-1])
            q1,q3=np.percentile(data,q=[25,75])
            lb = q1-r*(q3-q1)
            ub = q3+r*(q3-q1)
            return [i for i in data if lb<i<ub]
        else:
            if method.startswith('stdev'):
                r = float(method.split('-')[-1])
            else:
                r=3
            sig=np.std(data)
            mean=np.mean(data)
            return [i for i in data if (mean-r*sig)<i<(mean+r*sig)]

    def ifexist(self,name):
        if path.isfile(path.join(savefolder,name)):
            print('File {} already exist, add +.'.format(name))
            return self.ifexist(''.join(name.split('.')[0:-1]+['+.']+[name.split('.')[-1]]))
        else:
            return path.join(savefolder,name)


class Chip(DataMixin):
    """
    Class holding data of a chip.
    after init, have top, mid , bot attribute for Top middle and bottom electrodes.
    after electrode aggregate, chip data can be aggregated to single number hold in top.
    any operation on this class is making a new object without affecting original.
    """
    def __init__(self,top,mid=None,bot=None):
        self.top=top
        self.mid=mid
        self.bot=bot
        self.data=[top,mid,bot]
    def __bool__(self):
        return any(self.data)
    def __repr__(self):
        return "Chip[T:{},M:{},B:{}]".format(*self.data)
    def __iter__(self):
        for i in self.data:
            yield i
    def aggregate(self,method='avg',errors='ignore'):
        """
        method to aggregate electrode data on a chip.
        """
        data=[i.mb for i in self.data if i.mb]
        if errors=='delete':
            if len(data)!=len(self.data):
                return Chip(None)
        return Chip(self._convert(data,method))

class Eode(DataMixin):
    """
    Class holding data of a electrode.
    have mb and aq attribute for MB and AQ signal, stored in a list.
    any operation on this class is making a new object without affecting original.
    """
    def __init__(self,mb,aq=None):
        self.mb=mb
        self.aq=aq
    def __bool__(self):
        return bool(sum(self.mb))
    def __repr__(self):
        return "Eode:{:.2g}".format(np.mean(self.mb))
    def aggregate(self,mb='avg',aq='avg',aqnorm=False):
        """
        method to aggregate electrode data
        aggregate can be avg or number for picking a point.
        """
        mb=self._convert(self.mb,mb)
        aq=self._convert(self.aq,aq) if self.aq else 0
        # to decide if mb and aq have problems in converting to normal numbers.
        mb = 0 if np.isnan(mb) else mb
        aq = 0 if np.isnan(aq) else aq
        eode = Eode(mb,aq).aqnorm(aqnorm) if aqnorm else Eode(mb,aq)
        return eode

    def aqnorm(self,method='div'):
        """
        holds method for aq normalize.
        default is 'divide': divide mb signal by aq signal
        'exp': mb signal ** aq signal.
        'aq': use aq siganl only.
        """
        method=method if isinstance(method,str) else 'div'
        if method=='div':
            if self.aq==0:
                return Eode(0,None)
            return Eode(self.mb/self.aq,None)
        elif method=='exp':
            return Eode(self.mb**self.aq,None)
        elif method=='aq':
            return Eode(self.aq,None)
        else:
            return self

class Data(DataMixin):
    """
    Data holding class for chip data.
    initialize by providing csv filename and MB signal counts.
    after init, have attributes df, raw,
    df: dict of DataFrame holding data for each concentration point
    raw: dict of Chip class.
    after zip, will collapse all chip and electrode data based on provided methods into
    a single data point measurement for each concentration point.
    the zipped value is stored in .data attribute.
    zip again will refresh .data with new data.
    """
    def __init__(self,filename,mb=5):
        """
        filename: the .csv file holding data.
        have 1 row for concentration, followed by all chip electrode data at that concentration.
        electrode data should be arraged as T/M/B order on a single chip, and repeat chip data.
        vertically, should have MB data on top and AQ data below if applicable.
        specify mb to indicate how many rows are for MB signal. default first 5 data points are for MB.
        """
        global savefolder
        filefolder = path.dirname(filename)
        savefolder= path.join(filefolder,'analysis')
        if not path.isdir(savefolder):
            makedirs(savefolder)
        with open(filename) as f:
            data=csv.reader(f)
            r={}
            key=None
            content=[]
            for i in data:
                judge=[_ for _ in i if _!='']
                if len(judge)==1:
                    if key is not None:r[key]=pd.DataFrame(content).fillna(0)
                    key = float(judge[0])
                    r[key]=None
                    content=[]
                    continue
                if any(i): content.append(pd.to_numeric(i,errors='coerce'))
            r[key]=pd.DataFrame(content).fillna(0)
        self.df=r
        self.conc=sorted(list(r.keys()))
        if mb:self.organizedata(mb=mb)

    def organizedata(self,mb):
        """
        parse imported dataframe into raw dict.
        {conc:list of chips}
        chips: chip class holding mb and aq signal.
        """
        data = {}
        for i in self.conc:
            data[i]=[]
            df = self.df[i]
            for j in range(0,len(df.columns),3):
                chip = []
                for k in range(3):
                    aq=df.iloc[mb:,j+k].tolist() if mb<df.shape[0] else None
                    chip.append(Eode(df.iloc[0:mb,j+k].tolist(),aq))
                chip=Chip(*chip)
                if chip:data[i].append(chip)
        self.raw=data
        return self

    def zip(self,mb='avg',aq='avg',aqnorm=False,chip='avg',errors='ignore',inverse=False,**kwargs):
        """
        aggregate chip and electrode data by different methods.
        mb: method to aggregate mb signal on a electrode.
        aq: method to aggregate aq signal on a electrode.
        aqnorm: method to apply aq norm. if use aq, will use AQ raw signal to calculate.
        chip: method to aggregate electrode signal on a chip. can be T/M/B for top middle or bottom chip.
        each zip will refresh the data attribute of Data class. this is the converted data.
        """
        data={conc:[Chip(*[eode.aggregate(mb,aq,aqnorm) for eode in _chip]).aggregate(chip,errors).top
                    for _chip in chips] for conc,chips in self.raw.items()}
        if errors == 'delete':
            self.data={i:[_ for _ in j if _!=None] for i,j in data.items()}
        else:
            self.data=data
        if inverse:self.inverse()

        self.log=dict(mb=mb,aq=aq,aqnorm=aqnorm,chip=chip,errors=errors,inverse=inverse)

        return self

    def chipcount(self):
        return {i:len(j) for i,j in self.raw.items()}

    def inverse(self):
        """
        flip the sign of data.
        """
        for i in self.data:
            self.data[i]=[-k for k in self.data[i]]

    def analyze(self,range=None,xT='',yT='',**kwargs):
        """
        return an analyze object.
        """
        log=self.log.copy()
        log.update(xT=xT,yT=yT,range=range)
        return Analyzer(self.data,range=range,xT=xT,yT=yT,log=log)

class Analyzer(DataMixin):
    transform={'log':np.log10,'log_r':lambda x:10**x,'':lambda x:x,
                '_r':lambda x:x,'no':lambda x:x,'no_r':lambda x:x}
    def __init__(self,data,range=None,xT='',yT='',transform=True,log=None):
        ## temporary solution for log transformation to remove 0 x.
        if xT=='log': range=[0.001,1e6]

        if range:
            data = {i:j for i,j in data.items() if range[0]<=i<=range[1]}
        self.rawdata=data
        if transform:
            if xT: data={self.transform[xT](k):i for k,i in data.items()}
            if yT: data={k:[self.transform[yT](_) for _ in i] for k,i in data.items()}
        self.data=data
        self.x,self.y=np.array([i for i in self.data]),[np.array(j) for i,j in self.data.items()]
        self.xT=xT
        self.yT=yT
        self.xTt=self.transform[xT]
        self.xTr=self.transform[xT+'_r']
        self.yTt=self.transform[yT]
        self.yTr=self.transform[yT+'_r']
        self.log=log

    def savelog(self,save=True):
        result=['Data Processing Parameters']
        for k,i in self.log.items():
            result.append("{:>10} : {}".format(k,i))
        if save:
            txt_save = self.ifexist('Data_Ana_Log.txt')
            with open(txt_save,'wt') as f:
                f.write('\n'.join(result))
        return result

    def remove(self,key,idx):
        data = self.data.copy()
        data[key]=self.data[key].copy()
        data[key].pop(idx)
        if len(data[key])==0: data.pop(key)
        return np.array([i for i in data]),[np.array(j) for i,j in data.items()]

    def fit(self,multiset=True,method='linear',data=None):
        if data:
            x,y=data
        else:
            x,y=self.x,self.y
        para=fb.fitting_data(x,y,method,multi_set=multiset)
        rsquare=fb.r_squared_calc(self.x,self.y,para[0],method)
        return (*para,rsquare)

    def filter(self,range=None):
        return Analyzer(self.rawdata,range=range,xT=self.xT,yT=self.yT)

    def residuals(self,method='linear',resample=False,multiset=True,resi=None,**kwargs):
        if resi: return resi
        resi={i:[] for i in self.data}
        if not resample: para = self.fit(multiset=multiset, method=method)[0]
        for i,j in self.data.items():
            for idx,k in enumerate(j):
                if resample:
                    removed=self.remove(i,idx)
                    para=self.fit(multiset=multiset,method=method,data=removed)[0]
                func=getattr(fb,method+'_r')
                result=self.xTr(func(k,**para))
                resi[i].append(result-self.xTr(i))
        return {self.xTr(i):k for i,k in resi.items()}

    def cv_cacu(self,data):
        std=np.std(data,ddof=1)
        avg = np.mean(data)
        avg=max(0.00000001,avg)
        cv=min(100*std/avg,999)
        return cv

    def cv_calculator(self,conc_dict,format=True,correct=False):
        if correct:
            cv_dict = {i:self.cv_cacu([_+i for _ in j]) for i,j in conc_dict.items()}
        else:
            cv_dict = {i:self.cv_cacu(j) for i,j in conc_dict.items()}
        if format:
            data=sorted([i for i in cv_dict.items()],key=(lambda x:x[0]))
            df=pd.DataFrame({'Conc.':[i[0] for i in data],'CV':[i[1] for i in data]})
            return df
        else:
            return cv_dict

    def ci_cacu(self,data):
        stderror= np.std(data)/np.sqrt(len(data))
        stderror = max(1e-9,stderror)
        lb,up=t.interval(0.95,df=len(data)-1,loc=np.mean(data),scale=stderror)
        return lb,up

    def CI_calculator(self,format=True,conc_dict=None):
        _data=conc_dict
        ci_dict = {i:self.ci_cacu(j) for i,j in _data.items()}
        if format:
            re=sorted([i for i in ci_dict.items()],key=(lambda x:x[0]))
            df=pd.DataFrame({'Conc.':[i[0] for i in re],'95% CI lower':[i[1][0] for i in re],'95% CI upper':[i[1][1] for i in re]})
            return df
        else:
            return ci_dict

    def mean(self,format=True,**kwargs):
        r={self.xTr(i):np.mean(j) for i,j in self.data.items()}
        if format:
            data=sorted([i for i in r.items()],key=(lambda x:x[0]))
            df=pd.DataFrame({'Conc.':[i[0] for i in data],'Mean':[i[1] for i in data]})
            return df
        else:
            return r

    def mean_CV(self,format=True,**kwargs):
        converted={self.xTr(i):k for i,k in self.data.items()}
        return self.cv_calculator(conc_dict=converted,format=format)

    def mean_CI(self,format=True,**kwargs):
        converted={self.xTr(i):k for i,k in self.data.items()}
        return self.CI_calculator(conc_dict=converted,format=format)

    def predict_CV(self,format=True,resi=None,**kwargs):
        resi=resi or self.residuals(**kwargs)
        return self.cv_calculator(conc_dict=resi,format=format,correct=True)

    def predict_CI(self,format=True,resi=None,**kwargs):
        resi=resi or self.residuals(**kwargs)
        predict = {i:[_+i for _ in j] for i,j in resi.items()}
        return self.CI_calculator(conc_dict=predict,format=format)

    def predict(self,resi=None,**kwargs):
        resi=resi or self.residuals(**kwargs)
        return {i:[i+_ for _ in j] for i,j in resi.items()}

    def predict_mean(self,resi=None,format=True,**kwargs):
        resi=resi or self.residuals(**kwargs)
        r = {i:i+np.mean(j) for i,j in resi.items()}
        if format:
            data=sorted([i for i in r.items()],key=(lambda x:x[0]))
            df=pd.DataFrame({'Conc.':[i[0] for i in data],'Mean':[i[1] for i in data]})
            return df
        else:
            return r

    def plot_scatter(self,raw,ax=None,save=False,ylabel='None',log=False,**kwargs):
        raw_x=[self.xTt(i) for i,j in raw.items() for k in j]
        raw_y=[k for i,j in raw.items() for k in j]
        if ax==None:fig,ax = plt.subplots(figsize=(8,6))
        ax.plot(raw_x,raw_y,'bx')
        xs=sorted(list(set(raw_x)))
        ax.set_xticks(xs)
        ax.set_xticklabels(["{:.3g}".format(self.xTr(i)) for i in xs])
        ax.set_title('Residual plot')
        ax.set_xlabel('Fibrinogen concentration (nM)')
        ax.set_ylabel(ylabel)
        if log:ax.set_xscale('log')
        if save:
            save=save+plotbackend if isinstance(save,str) else "Unnamed_scatter_plot"+plotbackend
            save=self.ifexist(save)
            plt.tight_layout()
            plt.savefig(save)
            plt.clf()
            plt.close('all')

    def plot_resi(self,ax=None,save=False,ylabel='Predict Fib - True Fib /nM',log=False,resample=True,**kwargs):
        resi=self.residuals(resample=resample,**kwargs)
        if save:
            save=save if isinstance(save,str) else "RESI_pFib-tFib"
        self.plot_scatter(resi,ax=ax,save=save,ylabel=ylabel,log=log)

    def plot_fit(self,ax=None,multiset=True,save=False,method='linear',log=False,**kwargs):
        xlow = min(self.x) - 0.1*(max(self.x) - min(self.x))
        xhigh =max(self.x) + 0.1*(max(self.x) - min(self.x))
        x_ = (np.linspace(xlow,xhigh,100))
        func=getattr(fb,method)
        p,lp,up,r_2 = self.fit(multiset=multiset,method=method)
        if ax==None:fig,ax = plt.subplots(figsize=(8,6))
        raw_x=np.array([i for i,j in self.data.items() for k in j])
        raw_y=np.array([k for i,j in self.data.items() for k in j])
        ax.plot(x_,func(x_,**p),'g-')
        # if method=='linear':
        ax.plot(x_,func(x_,**lp),'r:',x_,func(x_,**up),'r:')
        ax.plot(raw_x,raw_y,'bx')
        _p=['{}:{:.4g}'.format(i,j) for i,j in p.items()]
        # locs,_=plt.xticks()
        ax.set_xticks(self.x)
        ax.set_xticklabels(["{:.3g}".format(self.xTr(i)) for i in self.x])
        ax.set_title('{} fit, {}; r2={:.4g}'.format(method,'; '.join(_p),r_2))
        ax.set_xlabel('Fibrinogen concentration (nM)')
        ax.set_ylabel('Raw signal {}'.format('('+self.yT+')'))
        if log: ax.set_xscale('log')
        if save:
            plt.tight_layout()
            save=save+plotbackend if isinstance(save,str) else 'FIT_{}_r{:.2f}'.format(method,r_2)+plotbackend
            save=self.ifexist(save)
            plt.savefig(save)
            plt.clf()
            plt.close('all')

    def shuffle(self,n=100,seed=42,**kwargs):
        np.random.seed(seed)
        for i in range(n):
            temp={i:list(np.random.choice(j,len(j))) for i,j in self.data.items()}
            yield temp

    def bootstrap(self,stats=None,size=1000,**kwargs):
        """
        stats is a list of method names to bootstrap.
        """
        kwargs.update(format=False)
        task=partial(workfunc,kw=kwargs,st=stats,xT=self.xT,yT=self.yT)
        # """test"""
        # workload=list(self.shuffle(size))
        # result = fb.poolwrapper(task,workload,total=size,showprogress=True)
        # return workload,result
        result = fb.poolwrapper(task,self.shuffle(n=size,**kwargs),total=size,showprogress=True)
        # summary = {i:[] for i in result[0].keys()}
        # summary_2 = {i:[] for i in result[0].keys()}
        total_result = dict.fromkeys(stats)
        for stat in stats:
            summary = {i:[] for i in result[0][stat].keys()}
            summary_2 = {i:[] for i in result[0][stat].keys()}
            for i in result:
                for j,d in i[stat].items():
                    if 'CI' in stat:
                        summary[j].append(d[0])
                        summary_2[j].append(d[1])
                    else:
                        if isinstance(d,list):
                            summary[j].extend(d)
                        else:
                            summary[j].append(d)
            if 'CI' in stat:
                summary = {"{:.3g}_LOW".format(k):i for k,i in summary.items()}
                summary_2 = {"{:.3g}_UP".format(k):i for k,i in summary_2.items()}
                summary.update(summary_2)
                # total_result[stat]=summary
                # return summary
            # else:
            total_result[stat]=summary
                # return summary

        return total_result

    def plot_bootstrap(self,save=False,stats=None,cumu=False,rmoutlier=True,shareax=False,**kwargs):
        possible_stats=['mean','mean_CI','mean_CV','residuals','predict','predict_CI','predict_CV','predict_mean',]
        if stats=='all':
            stat_list=possible_stats
        else:
            stat_list=[i for i in possible_stats if stats in i]
        repeats=kwargs.get('size',1000)
        total_result = self.bootstrap(stats=stat_list,**kwargs)
        for stat in stat_list:
            result = total_result[stat]
            if rmoutlier:
                for i in result.keys():
                    result[i] = self.rmoutlier(result[i],rmoutlier)
            if shareax:
                fig,_ = plt.subplots(figsize=(8,6))
                axes = [_]*len(result)
                _.set_yscale('log')
            else:
                panel = (max(2,len(result.keys())//4+bool(len(result.keys())%4)),4)
                fig,axes = plt.subplots(*panel,figsize=(12,2.5*panel[0]))
                axes=[i for k in axes for i in k]
            fig.suptitle('Bootstrap {}, repeat={:.1E}'.format(stat,repeats),size=16)
            for ax,(conc,resi),n in zip(axes,result.items(),self.samplesize()*2):
                title= conc if isinstance(conc,str) else '{:.2f}'.format(conc)
                labelcorrector=n if stat in ['residuals'] else 1
                ax.set_title("{}, N={}, {:.1f}%".format(title,n,100*len(resi)/(labelcorrector*repeats)))
                label=conc if isinstance(conc,str) else "{:.3g}".format(conc)
                ax.hist(resi,bins=500,histtype='step',density=True,cumulative=cumu,label=label)
                ax.set_xlabel('{}'.format(stat))
                ax.set_ylabel('Frequency')
                if shareax:ax.legend()
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                to_save = save+plotbackend if isinstance(save,str) else "BS_{}".format(stat)+plotbackend
                to_save = self.ifexist(to_save)
                plt.savefig(to_save)
                plt.clf()
                plt.close('all')
            else:
                plt.show()
                plt.clf()

    def samplesize(self):
        return [len(self.data[i]) for i in self.x]

    def analyze(self,multiset=True,method='linear',save=False,range=None,resample=True,**kwargs):
        if range: self=self.filter(*range)
        fig,axes = plt.subplots(1,2,figsize=(12,5))
        fig.suptitle('Fit & Residual, N={}'.format(self.samplesize()),size=16)
        self.plot_fit(ax=axes[0],multiset=multiset,method=method,**kwargs)
        self.plot_resi(ax=axes[1],method=method,resample=resample,**kwargs)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        result = '\n'.join(["Raw signal Mean:",str(self.mean()),'Raw signal CV:',str(self.mean_CV()),
                            "Predicted Fib Mean", str(self.predict_mean()),'Predicted Fib CV:',
                            str(self.predict_CV(method=method,resample=True)),'Predicted Fib CI:',
                            str(self.predict_CI(method=method,resample=True))])
        if save:
            save=save+plotbackend if isinstance(save,str) else 'ANA_{}_{}'.format(method,multiset*'ms')+plotbackend
            save =self.ifexist(save)
            plt.savefig(save)
            plt.clf()
            plt.close('all')
            txt_save = self.ifexist('Data_Analysis.txt')
            with open(txt_save,'wt') as f:
                f.write(result)
        else:
            plt.show()
            plt.clf()
            print(result)
