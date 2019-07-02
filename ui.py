from STAT import Data

def separator(n=1):
    for i in range(n):
        print('='*40)
    print('')


def load_file():
    filelist=input('Drop Raw data files: \n\n')
    filename=filelist.strip().replace('\ ',' ')
    separator()
    print('Files entered:\n',filename)
    separator()
    mb=input('MB points: (default 5) \n')
    mb = int(mb) if mb else 5
    data=Data(filename,mb)
    print('File loaded.\n')
    separator()
    return data

def enter_para(data):
    cycle=True
    while cycle:
        print("Enter Data processing parameter:\n")
        mb=input('MB method (avg, sd_rmo, p_rmo, mid): (default=avg)\n') or 'avg'
        aq=input('AQ method (avg, sd_rmo, p_rmo, mid): (default=avg)\n') or 'avg'
        aqnorm = input('AQ norm. method (div, exp, aq): (defalut=no)\n') or 'no'
        chip=input('Chip method (avg, T/M/B,): (default=avg)\n') or 'avg'
        inverse=input('Flip sign of data (y/n)?(default=n)\n') or 'n'
        xT=input("Transform X values? (log,) : (default=no)\n") or 'no'
        yT=input("Transform Y values? (log,) : (default=no)\n") or 'no'
        range=input("Enter X value range (1-99)? (default=all) \n") or 'all'
        result = dict(mb=mb,aq=aq,aqnorm=aqnorm,chip=chip,inverse=inverse,)
        transform =dict(xT=xT,yT=yT,range=range)
        separator()
        display=['Data Processing Parameters']
        display.extend(["{:>10} : {}".format(k,i) for k,i in result.items()])
        display.extend(["{:>10} : {}".format(k,i) for k,i in transform.items()])
        print('\n'.join(display))
        separator()
        ans = input("Paramters correct? Hit Enter to continue, n to re-Enter.")
        if ans!='n':
            cycle=False
        separator()
    result['inverse']=bool(result['inverse']!='n')
    transform['range'] =[float(range.split('-')[0]),float(range.split('-')[1])] if range!='all' else None
    analysis(data,result,transform)




def analysis(rawdata,result,transform):
    data=rawdata.zip(**result).analyze(**transform)
    cycle=True
    options={"1":"Plot Fit & Residuals", "2":"Bootstrap","3":"All", "0":"Quit Analysis"}
    while cycle:
        print('***'*10)
        print("\nAnalysis options:\n")
        print("\n".join(["{}: {}".format(k,i) for k,i in options.items()]))
        option = input("\nEnter analysis option: ")

        if option=="1":
            data.analyze(save=True)
        elif option=='2':
            data.plot_bootstrap(size=10000,stats='all',rmoutlier=True,cumu=False,resample=True,save=True)
        elif option=='3':
            data.analyze(save=True)
            data.plot_bootstrap(size=10000,stats='all',rmoutlier=True,cumu=False,resample=True,save=True)
        elif option=="0":
            break
        else:
            continue
        data.savelog()
        print("Analysis Done. ")
        separator()
    separator()
    ans = input("Re-Enter paramters to analyze this data? y to re-Enter.\n")
    if ans=='y':
        separator()
        enter_para(rawdata)

def main():
    print('\n'*1)
    cycle=True
    while cycle:
        data = load_file()

        enter_para(data)

        cont=input('Hit Enter to analyze new file, n to quit.')
        if cont=='n':
            cycle=False
        else:
            separator(2)
            print('\n\n')







if __name__=='__main__':
    main()
