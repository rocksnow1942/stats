

class Analyzer():
    def __init__(self,data):
        self.data=data





    def xy(x):
        return np.array([i[0] for i in x]),np.array([i[1] for i in x])

    def _fit(x,multiset=True,method='linear'):
        return fb.fitting_data(*xy(x),method,multi_set=multiset)

    def r_squared(x,fitpara,method='linear'):
        fp = fitpara[0]
        return fb.r_squared_calc(*xy(x),fp,method)

    def fit(x,multiset=True,method='linear'):
        para=_fit(x,multiset,method=method)
        return (*para,r_squared(x,para,method=method))

    def multiset(x):
        return fb.convert_x_y(*xy(x))[-1]

    def data_filter(data,lr,ur,log=False):
        if log:
            lr=lr+1e-9
            lr,ur=np.log10(lr),np.log10(ur)
        return [i for i in data if lr<=i[0]<=ur]

    def fibconc(y,para,method='linear',log=False):
        func=getattr(fb,method+'_r')
        if log:
            return 10**(func(y,**para))
        else:
            return func(y,**para)

    def residuals(raw,method='linear',log=False):
        resi=[]
        for i in raw:
            _raw=[_ for _ in raw if _!=i]
            para=fit(_raw,multiset=True,method=method)[0]
            x0=10**i[0] if log else i[0]
            resi.append((x0,fibconc(i[1],para,method=method,log=log)-x0))
        return resi

    def cv_cacu(data,avg=None):
        std=np.std(data,ddof=1)
        avg = avg if avg else np.mean(data)
        avg=max(0.00000001,avg)
        cv=min(100*std/avg,999)
        return cv

    def cv_calculator(resi,format=True,correct=False):
        conc_dict={i:[] for i in set([i[0] for i in resi])}
        for i in resi:
            conc_dict[i[0]].append(i[1])
        if correct:
            cv_dict = {i:cv_cacu([_+i for _ in j]) for i,j in conc_dict.items()}
        else:
            cv_dict = {i:cv_cacu(j) for i,j in conc_dict.items()}
        if format:
            data=sorted([i for i in cv_dict.items()],key=(lambda x:x[0]))
            df=pd.DataFrame({'Conc.':[i[0] for i in data],'CV':[i[1] for i in data]})
            return df
        else:
            return cv_dict

    def _cv_calculator(resi,format=True,correct=False):
        conc_dict={i:[] for i in set([i[0] for i in resi])}
        for i in resi:
            conc_dict[i[0]].append(i[1])
        if correct:
            cv_dict = {i:cv_cacu([_+i for _ in j]) for i,j in conc_dict.items()}
        else:
            cv_dict = {i:cv_cacu(j) for i,j in conc_dict.items()}
        return cv_dict

    def ci_cacu(data):
        stderror= np.std(data)/np.sqrt(len(data))
        lb,up=t.interval(0.95,df=len(data)-1,loc=np.mean(data),scale=stderror)
        return lb,up

    def CI_calculator(data,format=True):
        conc_dict={i:[] for i in set([i[0] for i in data])}
        for i in data:
            conc_dict[i[0]].append(i[1])
        ci_dict = {i:ci_cacu(j) for i,j in conc_dict.items()}
        if format:
            re=sorted([i for i in ci_dict.items()],key=(lambda x:x[0]))
            df=pd.DataFrame({'Conc.':[i[0] for i in re],'95% CI lower':[i[1][0] for i in re],'95% CI upper':[i[1][1] for i in re]})
            return df
        else:
            return ci_dict

    def residuals_CI(raw,log=False,format=True):
        resi=residuals(raw,log=log)
        ci = [(i,i+j) for i,j in resi]
        return CI_calculator(ci,format=format)

    def residuals_cv(raw,log=False,format=True):
        resi=residuals(raw,log=log)
        return cv_calculator(resi,format=format,correct=True)

    def plot_scatter(raw,save=False,ylabel='None'):
        xlow = min(xy(raw)[0]) - 0.1*(max(xy(raw)[0]) - min(xy(raw)[0]))
        xhigh =max(xy(raw)[0]) + 0.1*(max(xy(raw)[0]) - min(xy(raw)[0]))
        fig,ax = plt.subplots(figsize=(8,6))
        plt.plot(*xy(raw),'bx')
        ax.set_title('residual')
        ax.set_xlim(xlow,xhigh)
        ax.set_xlabel('Fibrinogen concentration (nM)')
        ax.set_ylabel(ylabel)
        if save:
            plt.savefig(ylabel+'.svg')
        else:plt.show()

    def plot_resi(raw,log=False,method='linear',save=False,ylabel='Predict Fib - True Fib'):
        resi=residuals(raw,log=log,method=method)
        plot_scatter(resi,save=save,ylabel=ylabel)

    def plot_fit(raw,multiset=True,log=False,save=False,method='linear'):
        xlow = min(xy(raw)[0]) - 0.1*(max(xy(raw)[0]) - min(xy(raw)[0]))
        xhigh =max(xy(raw)[0]) + 0.1*(max(xy(raw)[0]) - min(xy(raw)[0]))
        x_ = np.linspace(xlow,xhigh,100)
        func=getattr(fb,method)
        p,lp,up,r_2 = fit(raw,multiset,method=method)
        fig,ax = plt.subplots(figsize=(8,6))
        raw_x,raw_y=xy(raw)
        x__=10**x_ if log else x_
        raw_x=10**raw_x if log else raw_x
        plt.plot(x__,func(x_,**p),'g-')
        if method=='linear':plt.plot(x__,func(x_,**lp),'r:',x__,func(x_,**up),'r:')
        plt.plot(raw_x,raw_y,'bx')
        _p=['{}:{:.3f}'.format(i,j) for i,j in p.items()]
        ax.set_title('{} ; r2 = {:.4f}'.format('; '.join(_p),r_2))
        ax.set_xlabel('{} Fibrinogen concentration (nM)'.format(log*'Log'))
        ax.set_ylabel('{} Raw signal'.format(log*'Log'))
        if log:
            ax.set_xscale('log')
        if save:
            plt.savefig('{}_r2:{:.3f}.svg'.format('; '.join(_p),r_2))
        else:plt.show()

    def analyze(file,low=0,up=1e9,multiset=True,method='linear',log=False,save=False):
        raw,lograw=readfile(file)
        data=lograw if log else raw
        data=data_filter(data,low,up,log)
        plot_fit(data,multiset=multiset,log=log,save=save,method=method)
        plot_resi(data,log=log,save=save,method=method)
        print('Predicted Fib CV:')
        print(residuals_cv(data,log=log))
        print('Predicted Fib CI:')
        print(residuals_CI(data,log=log))
