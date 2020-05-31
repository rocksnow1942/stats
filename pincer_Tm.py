from mymodule import NUPACK
from mymodule import RNAstructure
import primer3
import numpy as np
import matplotlib.pyplot as plt
import random
from mymodule import revcomp,ProgressBar
from itertools import permutations

mv_conc=150
dv_conc = 4
dntp_conc = 0
dna_con = 1000
conc = (mv_conc,dv_conc,dntp_conc,dna_con)


def calc_dG_Tm_NuPack(sequences=[],concentrations=[1e-6,1e-9],Tm = list(range(12,46))):
    'sequences in order of solution,capture'
    conc = []
    dG = []
    Q = []
    for t in Tm:
        r = NUPACK.complexes(sequences,concentrations,maxcofoldstrand=2,T=t,sodium=0.15,magnesium=0.004)
        for k,i in r.items():
            if i[0]=='S1 + S2':
                dG.append(i[4])
                Q.append(i[1])
                conc.append(i[2])
                break
    return np.array(Tm),np.array(conc),np.array(dG),np.array(Q)


def randomseq(length=8):
    np.random.seed(42)
    while 1:
        yield ''.join(np.random.choice(list('ATCG'),size=length))

def shuffleseq(seq=''):
    np.random.seed(42)
    seq = list(seq)
    while 1:
        np.random.shuffle(seq)
        yield ''.join(seq)


def plot_comparison(pincers=[],captures=[],names=[],concentrations=[1e-6,1e-9]):
    Pin = 'GGCATTGCGACTAGGTTGGGTAGGGTGGTGTCGCTTTTTTTTTTAGATTCTC'
    Cap = 'GAGAATCT'
    if not captures:
        captures = [revcomp(i) for i in pincers]
    if not names:
        names = ['Pincer'] + [f'Seq-{i+1}' for i in range(len(pincers))]
    pincers.insert(0,Pin)
    captures.insert(0,Cap)
    concs=[]
    dGs = []
    for p, c in zip(pincers,captures):
        Tm,conc,dG,Q = calc_dG_Tm_NuPack([p,c],concentrations,Tm=range(12,47,1))
        concs.append(conc)
        dGs.append(dG)
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    ax,ax2 = axes
    for n,c,g in zip(names,concs,dGs):
        ax.plot(Tm,g,label=n)
        ax2.plot(Tm,c/concentrations[1]*100,label=n)
    ax.legend()
    ax.set_xlabel('Temperature')
    ax.set_ylabel('deltaG / kcal/mol')
    ax.set_title('deltaG - Temperature')
    ax2.legend()
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Percent binding %')
    ax2.set_title('%Binding - Temperature')
    plt.tight_layout()
    plt.show()
    return fig


Pin = 'GGCATTGCGACTAGGTTGGGTAGGGTGGTGTCGCTTTTTTTTTTAGATTCTC'
PI = 'AGATTCTC'
Cap = 'GAGAATCT'

IPT = 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTCAGTACTT'
IP = 'CAGTACTT'
IPCap = 'AAGTACTG'

# calculate pincer binding at 12 deg.
Tm,concP,dG,Q = calc_dG_Tm_NuPack([Pin,Cap],Tm=[12])

# generate random 8 n.t. sequences
results = []
for seq in randomseq(8):
    Tm,conc,dG,Q = calc_dG_Tm_NuPack([seq,revcomp(seq)],Tm=[12])
    results.append((seq,conc[0],dG[0]))
    if len(results)>1000:
        break
higher = sorted (list(filter(lambda x:x[1]-concP[0]>=0,results )), key = lambda x: abs(x[1]-concP[0]) )
f= plot_comparison(pincers=[higher[i][0] for i in [0,150,350,600,700]],)
f.savefig('Higher Than pincer.svg')


# plot those sequence with 44n.t. T at 5' end.
pickedhigher = [higher[i][0] for i in [0,150,350,600,700]]
f = plot_comparison(pincers=['T'*44 + i for i in pickedhigher],captures=[revcomp(i) for i in pickedhigher])
f.savefig()



# generate cap 8 permutation sequences
cap8permut = []
for s in permutations(Cap):
    seq = ''.join(s)
    Tm,conc,dG,Q = calc_dG_Tm_NuPack([seq,revcomp(seq)],Tm=[12])
    cap8permut.append((seq,conc[0]))
