from scipy.optimize import curve_fit
import numpy as np
from scipy.stats import t
import psutil
import multiprocessing
import tqdm
supported_fit_method = ['kd','ic_50','ric_50','kd_with_depletion','kd_ns','kd_ns_w_depletion','DR_4PL','DR_5PL','linear','various']



def poolwrapper(task,workload,initializer=None,initargs=None,chunks=None,total=None,desc='Progress:',showprogress=False):
    workerheads=psutil.cpu_count(logical=False)
    worker=multiprocessing.Pool(workerheads,initializer,initargs)
    total = total or len(workload)
    chunksize= int(total//chunks)+1 if chunks else 1
    if showprogress:
        result = []
        for _ in tqdm.tqdm(worker.imap(task,workload,chunksize),desc=desc,total=total):
            result.append(_)
    else:
        result=worker.imap(task,workload,chunksize)
    worker.close()
    worker.join()
    worker.terminate()
    return result



def r_squared_calc(x,y,fit_para,fit_method):
    """
    give the fit method and normal scale fit_para dict, calculate r_squared.
    """
    x_o_tofit=np.array([x[i] for i,j in enumerate(y) for k in j])
    y_o_tofit=np.array([k for i,j in enumerate(y) for k in j])
    func,_b,_a = fit_method_fetcher(fit_method)
    residuals = y_o_tofit-func(x_o_tofit,**fit_para)
    ss_residual = np.sum(residuals**2)
    ss_total = np.sum((y_o_tofit-np.mean(y_o_tofit))**2)
    r_squared = 1- ss_residual/ss_total
    return r_squared

def kd_r(y,kd,Fmax,Fmin,**kwarg):
    return 10**kd/(Fmax-Fmin)/(1/(y-Fmin)-1/(Fmax-Fmin))

def kd(x,kd,Fmax,Fmin,**kwargs):
    return ((Fmax-Fmin)*x/(x+10**kd))+Fmin

def exp(x,a,k):
    return a*(x**k)

def exp_r(y,a,k):
    return (y/a)**(1/k)

def log(x,a,b):
    return a*np.log10(x)+b
def log_r(y,a,b):
    return 10**((y-b)/a)

def kd_ns(x,kd,ns,Fmax,Fmin,**kwargs):
    return ((Fmax-Fmin)*x/(x+10**kd))+Fmin+ns*x

def ic_50(x,ic_50,Fmax,Fmin,**kwargs):
    return Fmax - (Fmax-Fmin)*x/(x+10**ic_50)

def kd_with_depletion(x,kd,a_0,Fmax,Fmin,**kwargs):
    kd = 10**kd
    a_0 = 10**a_0
    ab =a_0-((a_0-kd-x)+np.sqrt((kd+x)**2+a_0**2+2*kd*a_0-2*x*a_0))*0.5
    return (Fmax-Fmin)*ab/a_0 +Fmin

def kd_ns_w_depletion(x,kd,ns,a_0,Fmax,Fmin,**kwargs):
    kd = 10**kd
    a_0 = 10**a_0
    ab =a_0-((a_0-kd-x)+np.sqrt((kd+x)**2+a_0**2+2*kd*a_0-2*x*a_0))*0.5
    return (Fmax-Fmin)*ab/a_0 +Fmin +ns*x

def DR_4PL(x,ec_50,Fmax,Fmin,Hill,**kwargs):
    ec_50 = 10**ec_50
    return Fmax - (Fmax-Fmin)*(x**Hill)/(x**Hill+(ec_50)**Hill)

def DR_5PL(x,ec_50,Fmax,Fmin,Hill,S,**kwargs):
    """
    Five parameters logistic regression
    signal = Fmin + (Fmax-Fmin)*(X**(Hill*S))/(X**Hill + EC50**Hill*(2**(1/S)-1))**S
    """
    ec_50 = 10**ec_50
    denominator = (x**Hill+(ec_50)**Hill*(2**(1/S)-1))**S
    signal = Fmax - (Fmax-Fmin)*(x**(Hill*S))/denominator
    return signal

def linear_r(y,slope,b,**kwargs):
    return (y-b)/slope

def linear(x,slope,b,**kwargs):
    return slope*x+b

def ric_50(a_0, r_0, v_0, kd_r, kd_a,Fmax,Fmin,**kwargs):
    kd_r = 10**kd_r
    kd_a = 10**kd_a
    r_0 = 10**r_0
    v_0 = 10**v_0
    a=kd_a-kd_r
    d=-v_0*kd_a*r_0**2
    result=np.array([])
    min_rv=min(r_0,v_0)
    for i in a_0:
        b=kd_r*r_0-2*kd_a*r_0+kd_r**2-kd_r*kd_a-kd_r*i+kd_r*v_0-kd_a*v_0
        c=r_0**2*kd_a+kd_r*kd_a*r_0-v_0*r_0*kd_r+2*v_0*r_0*kd_a+i*kd_r*r_0
        root=np.roots(np.array([a,b,c,d]))
        if root.dtype == 'complex128':
            real_root = [i.real for i in root if i.imag == 0]
            if len(real_root)==1:
                pass
            else:
                real_root=[0]
        else:
            real_root = [i for i in root if 0<i<min_rv ]
            if len(real_root)==1:
                pass
            else:
                real_root=[0]
        result=np.append(result,real_root)
    signal = result/r_0*(Fmax-Fmin) + Fmin
    return signal

def fit_method_fetcher(fit_method,**kwargs):
    m_=[1e-3,1e5]
    Fm_ = [0,2e6]
    def kwarg_get(kwargs,tag,default):
        res_ = default if kwargs.get(tag,default) == 'default' else kwargs.get(tag,default)
        if tag in ['kd','ec_50','ic_50','r_0','v_0','kd_r','kd_a','a_0']:
            res_ = [np.log10(i) if i != 0 else -3 for i in res_]
        return res_
    if fit_method == 'kd':
        func = kd
        bound_name = ['kd','Fmax','Fmin']
        bounds_= tuple(list(i) for i in zip(kwarg_get(kwargs,'kd',m_),kwarg_get(kwargs,'Fmax',Fm_),kwarg_get(kwargs,'Fmin',Fm_)))
    elif fit_method == 'DR_4PL':
        func = DR_4PL
        bound_name = ['ec_50','Fmax','Fmin','Hill']
        bounds_ = tuple(list(i) for i in zip(kwarg_get(kwargs,'ec_50',m_),kwarg_get(kwargs,'Fmax',Fm_),kwarg_get(kwargs,'Fmin',Fm_),kwarg_get(kwargs,'Hill',[0.01,2])))
    elif fit_method == 'DR_5PL':
        func = DR_5PL
        bound_name = ['ec_50','Fmax','Fmin','Hill','S']
        bounds_ = tuple(list(i) for i in zip(kwarg_get(kwargs,'ec_50',m_),kwarg_get(kwargs,'Fmax',Fm_),kwarg_get(kwargs,'Fmin',Fm_),kwarg_get(kwargs,'Hill',[0.01,2]),kwarg_get(kwargs,'S',[0.1,10])))
    elif fit_method == 'linear':
        func = linear
        bound_name = ['slope','b']
        bounds_ =  tuple(list(i) for i in zip(kwarg_get(kwargs,'slope',[-1e3,1e3]),kwarg_get(kwargs,'b',[-1e3,1e3])))
    elif fit_method == 'ic_50':
        func = ic_50
        bound_name = ['ic_50','Fmax','Fmin']
        bounds_ = tuple(list(i) for i in zip(kwarg_get(kwargs,'ic_50',m_),kwarg_get(kwargs,'Fmax',Fm_),kwarg_get(kwargs,'Fmin',Fm_)))
    elif fit_method == 'ric_50':
        func = ric_50
        bound_name = ['r_0','v_0','kd_r','kd_a','Fmax','Fmin']
        bounds_= tuple(list(i) for i in zip(kwarg_get(kwargs,'r_0',m_),kwarg_get(kwargs,'v_0',m_),kwarg_get(kwargs,'kd_r',m_),kwarg_get(kwargs,'kd_a',m_),kwarg_get(kwargs,'Fmax',Fm_),kwarg_get(kwargs,'Fmin',Fm_)))
    elif fit_method == 'kd_with_depletion':
        func = kd_with_depletion
        bound_name = ['kd','a_0','Fmax','Fmin']
        bounds_=tuple(list(i) for i in zip(kwarg_get(kwargs,'kd',m_),kwarg_get(kwargs,'a_0',m_),kwarg_get(kwargs,'Fmax',Fm_),kwarg_get(kwargs,'Fmin',Fm_)))
    elif fit_method == 'kd_ns':
        func = kd_ns
        bound_name = ['kd','ns','Fmax','Fmin']
        bounds_=tuple(list(i) for i in zip(kwarg_get(kwargs,'kd',m_),kwarg_get(kwargs,'ns',Fm_),kwarg_get(kwargs,'Fmax',Fm_),kwarg_get(kwargs,'Fmin',Fm_)))
    elif fit_method == 'kd_ns_w_depletion':
        func = kd_ns_w_depletion
        bound_name = ['kd','ns','a_0','Fmax','Fmin']
        bounds_=tuple(list(i) for i in zip(kwarg_get(kwargs,'kd',m_),kwarg_get(kwargs,'ns',Fm_),kwarg_get(kwargs,'a_0',m_),kwarg_get(kwargs,'Fmax',Fm_),kwarg_get(kwargs,'Fmin',Fm_)))
    elif fit_method == 'exp':
        func = exp
        bound_name = ['a','k']
        bounds_=tuple(list(i) for i in zip(kwarg_get(kwargs,'a',[-1e3,1e3]),kwarg_get(kwargs,'k',[0.001,3])))
    elif fit_method == 'log':
        func = log
        bound_name = ['a','b']
        bounds_=tuple(list(i) for i in zip(kwarg_get(kwargs,'a',[-1e3,1e3]),kwarg_get(kwargs,'b',[-1e3,1e3])))
    else:
        fit_method=='kd'
        func = kd
        bound_name = ['kd','Fmax','Fmin']
        bounds_= tuple(list(i) for i in zip(kwarg_get(kwargs,'kd',m_),kwarg_get(kwargs,'Fmax',Fm_),kwarg_get(kwargs,'Fmin',Fm_)))
    return func, bounds_,bound_name

def convert_x_y(x,y):
    conc_set = set(x)
    if len(x)!=len(conc_set):
        new_x = list(conc_set)
        new_y = [[] for i in range(len(new_x))]
        for i,j in zip(x,y):
            new_y[new_x.index(i)].append(j)
        multi_set = True
        y_mean = [np.mean(i) for i in new_y]
        temp = [np.std(i,ddof=1) if len(i)>1 else 0 for i in new_y]
        average_temp = np.mean([i for i in temp if i!=0])
        y_stdev = [average_temp if i==0 else i for i in temp]
    else:
        new_x = x
        y_mean = y
        y_stdev = 0
        multi_set = False
    return new_x, y_mean, y_stdev, multi_set

def fitting_data(x,y,fit_method,multi_set=True,**bounds):
    func,bounds_,bound_name = fit_method_fetcher(fit_method,**bounds)
    p_0 = [0.5*(i+j) for i,j in zip(bounds_[0],bounds_[1])]
    if multi_set:
        y,y_stdev = [np.mean(i) for i in y],[np.std(i,ddof=1) if len(i)>1 else 0 for i in y]
        average_std = np.mean([i for i in y_stdev if i!=0])
        y_stdev = [average_std if i==0 else i for i in y_stdev]

    else:
        x=[x[i] for i,j in enumerate(y) for k in j]
        y=[   k for i,j in enumerate(y) for k in j]
    try:
        freedom = max(1,len(x)-len(bound_name))
        lower_CI = []
        upper_CI = []
        if multi_set:
            fit_result,corv_=curve_fit(func, x, y,p0=p_0,sigma=y_stdev,bounds = bounds_,absolute_sigma=False) #
        else:
            fit_result,corv_=curve_fit(func, x, y,p0=p_0,bounds = bounds_,absolute_sigma=False)
        sigma = np.sqrt(np.diagonal(corv_))
        for i,j in zip(sigma,fit_result):
            C_interval = t.interval(0.95,freedom,j,i)
            lower_CI.append(C_interval[0])
            upper_CI.append(C_interval[1])
    except Exception as e:
        print(e)
        fit_result = [1]*len(bound_name)
        lower_CI = fit_result
        upper_CI = fit_result
    return dict(zip(bound_name, fit_result)),dict(zip(bound_name, lower_CI)),dict(zip(bound_name, upper_CI))

def log_para_coverter(fit_para,to_log=True):
    new_para = dict.fromkeys(fit_para,0.0)
    for key,item in fit_para.items():
        if key in ['kd','ec_50','ic_50','r_0','v_0','kd_r','kd_a','a_0']:
            if to_log:
                new_para[key]=np.log10(item)
            else:
                new_para[key]=10**item
        else:
            new_para[key]=item
    return new_para

def fit_CI_value(func,x_fit,fit_method,fit_para_CI,fit_para):
    lower_para = log_para_coverter(fit_para_CI['lower_CI'])
    upper_para = log_para_coverter(fit_para_CI['upper_CI'])
    fit_result = log_para_coverter(fit_para)
    if fit_method != 'linear':
        if 'kd' in fit_method:
            temp = lower_para['kd']
            lower_para['kd'] = upper_para['kd']
            upper_para['kd'] = temp
            if 'depletion' in fit_method:
                lower_para['a_0']=upper_para['a_0']=fit_result.get('a_0',None)
        elif fit_method=='ric_50':
            lower_para=upper_para=fit_result
        elif fit_method == 'ic_50':
            pass
        elif 'DR_' in fit_method:
            if fit_result['Fmax'] < fit_result['Fmin']:
                temp = lower_para['ec_50']
                lower_para['ec_50'] = upper_para['ec_50']
                upper_para['ec_50'] = temp
            else:
                pass
            lower_para['Hill']=upper_para['Hill']=fit_result.get('Hill',None)
            if fit_method == 'DR_5PL':
                lower_para['S']=upper_para['S']=fit_result.get('S',None)
        else:
            pass
        y_fit_lower_CI = func(x_fit,**lower_para)
        y_fit_upper_CI = func(x_fit,**upper_para)
    else:
        a = func(x_fit,**lower_para)
        b = func(x_fit,**upper_para)
        y_fit = np.stack((a,b),axis=0)
        y_fit_lower_CI = y_fit.min(axis=0)
        y_fit_upper_CI = y_fit.max(axis=0)
    return y_fit_lower_CI, y_fit_upper_CI
