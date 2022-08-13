import math
from re import A
import numpy as np

Nu_mfx = 3
Nu_mfy = 3

chromosomeSize = (Nu_mfx*2) + (Nu_mfy*2) + (Nu_mfx*Nu_mfy*3)

chromosome = [0, 0.5, 1, 0, 0.5, 1, .21, 1.21, .21, .21, .21, .21, 0, 10000, 0, 0, -1000, 0, 0, -1000, 0, 0, 10000, 0, 0, 0, 0, 0, -1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000]

size = 10

x = []
y = []
z = np.empty((size,size))


for i in range(0,size):
    x.append((i+1)/10)
    for j in range(0,size):

        y.append((j+1)/10)

        d_start = Nu_mfx+Nu_mfy
        mfx = []
        for e in range(0, Nu_mfx):
            if (chromosome[e+d_start] == 0):
                mfx.append(0)
            else:
                mfx.append(math.exp((-(x[i]-chromosome[e])**2)/(2*chromosome[e+d_start]**2)))

        mfy = []
        for e in range(0, Nu_mfy):
            if (chromosome[e+d_start+Nu_mfx] == 0):
                mfy.append(0)
            else:
                mfy.append(math.exp((-(y[j]-chromosome[e+Nu_mfx])**2)/(2*chromosome[e+d_start+Nu_mfx]**2)))

        inf = []
        for e in range(0, Nu_mfx):
            for u in range(0, Nu_mfy):
                inf.append(mfx[e]*mfy[u])
        
        p_start = (Nu_mfx+Nu_mfy)*2
        q_start = p_start + (Nu_mfx*Nu_mfy)
        r_start = q_start + (Nu_mfx*Nu_mfy)

        reg = []
        for e in range(0, Nu_mfx*Nu_mfy):
            reg.append(inf[e]*((chromosome[e+p_start]*x[i])+(chromosome[e+q_start]*y[j])+chromosome[e+r_start]))

        
        a = 0
        b = 0

        for e in range(0, Nu_mfx*Nu_mfy):
            b += inf[e]
            a += reg[e]
        
        z[i][j]=a/b