import visdom
import time
import numpy as np

'''
Borrowed from https://blog.csdn.net/LXX516/article/details/79019328
'''
 
class Visualizer(object):
    def __init__(self, env='default', port = 5678, **kwargs):
        self.vis = visdom.Visdom(server='202.120.39.167', port=port, env=env, **kwargs)
        self.index = {}         
    def plot_many_stack(self, d):
        '''
        self.plot('loss',1.00)
        '''
        name=list(d.keys())
        name_total=" ".join(name)
        x = self.index.get(name_total, 0)
        val=list(d.values())
        if len(val)==1:
            y=np.array(val)
        else:
            y=np.array(val).reshape(-1,len(val))
        #print(x)
        self.vis.line(Y=y,X=np.ones(y.shape)*x,
                    win=str(name_total),#unicode
                    opts=dict(legend=name,
                        title=name_total),
                    update=None if x == 0 else 'append'
                    )
        self.index[name_total] = x + 1 