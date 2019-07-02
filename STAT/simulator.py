import numpy as np
from .cv_analysis import Analyzer




class Simu(Analyzer):
    def __init__(self,conc=[0,0.25,0.5,0.75,1],sig=[100,122.5,145,167.5,190],
                std=[7,8.575,10.15,11.725,13.3],size=5):
        """
        default y = 90x+100; CV=7% on all points.
        """
        data = dict.fromkeys(conc)
        size = size if isinstance(size,list) else [size]*len(conc)
        assert len(conc)==len(std)==len(sig)==len(size), ("lenth not uniform")
        for c,s,d,si in zip(conc,sig, std, size):
            data[c]=list(np.random.normal(s,d,si))
        Analyzer.__init__(self,data)
