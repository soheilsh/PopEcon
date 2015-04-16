"""

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# == Define parameters == #
eta_a = 0.01       # Coefficient of technological growth in Agriculture
eta_m = 0.02       # Coefficient of technological growth in Manufacturing
delta = 0.1        # Power of Technological growth function
alpha = 0.1        # Coefficient of Agricultural consumption in Utility function
beta = 0.7         # Coefficient of Manufacturing consumption in Utility function
Aa_0 = 10          # Initial level of technology in Agriculture
Am_0 = 45          # Initial level of technology in Manufacturing
C_0 = 0.01         # Minimum level of Agricultural consumption
eps = 0.6          # Power of Manufacturing input in production function
tu = 0.1           # Time spent to raise an unskilled child
ts = 0.3           # Time spent to raise a skilled child
T = 200            # Time horizon
Tmax = 11          # Max Modeling Time
stepsize = 0.1     # Step size

# == Generate age matrix == #
u = np.zeros((20, T))           # Unskilled children
s = np.zeros((20, T))           # Skilled children
l = np.zeros((40, T))           # Unskilled adults (Total)
m = np.zeros((40, T))           # Skilled adults

# == Labor == #
mt = [0]*T          # Total skilled adults in Manufacturing
lt = [0]*T          # Total unskilled adults
lmt = [0]*T         # Total unskilled adults in Manufacturing 
lat = [0]*T         # Total unskilled adults in Agriculture 

# == Generate education matrix == #
gammau = np.zeros((20, T))
gammas = np.zeros((20, T))

# == Prices == #
pa = [0]*T
pm = [0]*T

# == Wages == #
wa = [0]*T
wm = [0]*T

# == Technology == #
B = [0]*T
Aa = [Aa_0]*T
Am = [Am_0]*T

# == Output == #
Ya = [0]*T
Ym = [0]*T

# == Consumption == #
ca = np.zeros((2, T))
cm = np.zeros((2, T))

# == Utility == #
w = np.zeros((2, T))

# == Result == #
result = np.zeros((2, T))
nopt = 0.1 + np.zeros((4, T))
nopt1 = 0.1 + np.zeros((4, T))
nopt2 = 0.1 + np.zeros((4, T))

# == Error == #
wr = [0]*T          # Children future wage ratio 
er = [0]*T
tol = 0.1

# Initializing
for i in range(20):
    u[i, 0] = 1.5
    s[i, 0] = 0.25
    l[i, 0] = 1
    m[i, 0] = 0.07
    l[i+20, 0] = 0.6
    m[i+20, 0] = 0.03    
    gammau[i, :] = 1-alpha-beta
    gammas[i, :] = 1-alpha-beta
mt[0] = sum(m[:,0])
lt[0] = sum(l[:,0])    
B[0] = (mt[0]/(mt[0] + lt[0]))**delta
lmt[0] = (((beta/alpha)/Aa[0]) * (1 - eps) * (Aa[0] * lt[0] - (mt[0] + lt[0]) * C_0))/(1 + (beta/alpha) * (1 - eps))
lat[0] = lt[0] - lmt[0]
Ya[0] = Aa[0] * lat[0]
Ym[0] = Am[0] * mt[0]**eps * lmt[0]**(1 - eps)
pa[0] = (Am[0]/Aa[0]) * (1-eps) * (mt[0]/lmt[0])**eps
wa[0] = pa[0] * Aa[0]
wm[0] = eps * Am[0] * (mt[0]/lmt[0])**(eps - 1)   
    
def util(n):
    for t in range(tx, tx + 60):
        if t == tx:
            if typ == 1:        #Unskilled
                nuu = n[0]
                nsu = n[1]
                nus = nopt[2, tx]
                nss = nopt[3, tx]
            else:              #Skilled
                nuu = nopt[0, tx]
                nsu = nopt[1, tx]
                nus = n[0]
                nss = n[1]
        else:
            nuu = nopt[0, t]
            nsu = nopt[1, t]
            nus = nopt[2, t]
            nss = nopt[3, t]
        gammau[0, t] = nuu * tu + nsu * ts
        gammas[0, t] = nus * tu + nss * ts       
        uu = nuu * u[19, t-1] 
        su = nsu * u[19, t-1]
        us = nus * s[19, t-1]
        ss = nss * s[19, t-1]
        u[0, t] = uu + us
        s[0, t] = su + ss
        u[1:20, t] = u[0:19, t-1]
        s[1:20, t] = s[0:19, t-1]
        l[0, t] = u[19, t-1] * (1 - gammau[0, t])
        m[0, t] = s[19, t-1] * (1 - gammas[0, t])             
        l[1:20, t] = l[0:19, t-1]
        m[1:20, t] = m[0:19, t-1]
        l[20, t] = l[19, t-1] / (1 - gammau[19, t-1])
        m[20, t] = m[19, t-1] / (1 - gammas[19, t-1])             
        l[21:40, t] = l[20:39, t-1]
        m[21:40, t] = m[20:39, t-1]
        gammau[1:20, t] = gammau[0:19, t-1]
        gammas[1:20, t] = gammas[0:19, t-1]
        mt[t] = sum(m[:,t])
        lt[t] = sum(l[:,t])
        B[t] = (mt[t]/(mt[t] + lt[t]))**delta
        Aa[t] = (1 + eta_a * B[t]) * Aa[t-1]
        Am[t] = (1 + eta_m * B[t]) * Am[t-1]
        lmt[t] = (((beta/alpha)/Aa[t]) * (1 - eps) * (Aa[t] * lt[t] - (mt[t] + lt[t]) * C_0))/(1 + (beta/alpha) * (1 - eps))
        lat[t] = lt[t] - lmt[t]
        Ya[t] = Aa[t] * lat[t]
        Ym[t] = Am[t] * mt[t]**eps * lmt[t]**(1 - eps)
        pa[t] = (Am[t]/Aa[t]) * (1-eps) * (mt[t]/lmt[t])**eps
        wa[t] = pa[t] * Aa[t]
        wm[t] = eps * Am[t] * (mt[t]/lmt[t])**(eps - 1)
    if typ == 1:        #Unskilled
        ca[typ-1, tx] = (wa[tx] * (1-gammau[0, tx]) + (beta/alpha) * pa[tx] * C_0)/((1+ (beta/alpha)) * pa[tx])
        cm[typ-1, tx]  = (beta/alpha) * pa[tx] * (ca[typ-1, tx] - C_0)
        w[typ-1, tx]  = -(alpha * np.log(ca[typ-1, tx] - C_0) + beta * np.log(cm[typ-1, tx] ) + (1 - alpha - beta) * np.log(n[0] * sum(wm[tx+20:tx+60]) + n[1] * sum(wa[tx+20:tx+60])))
    else:              #Skilled
        ca[typ-1, tx] = (wm[tx] * (1-gammas[0, tx]) + (beta/alpha) * pa[tx] * C_0)/((1+ (beta/alpha)) * pa[tx])
        cm[typ-1, tx]  = (beta/alpha) * pa[tx] * (ca[typ-1, tx] - C_0)
        w[typ-1, tx]  = -(alpha * np.log(ca[typ-1, tx] - C_0) + beta * np.log(cm[typ-1, tx] ) + (1 - alpha - beta) * np.log(n[0] * sum(wm[tx+20:tx+60]) + n[1] * sum(wa[tx+20:tx+60])))
    return w[typ-1, tx]

for i in range(2):
    for tx in range(1,Tmax):
        if er[tx] > tol:
            bnds = [(0, 0), (0, None)]
        else:
            if er[tx] < tol:
                bnds = [(0, None), (0, 0)]
            else:
                bnds = [(0, None), (0, None)]
        for typ in [1, 2]:
            x0 = nopt[(typ-1)*2:(typ-1)*2+2, tx]
            cons = ({'type': 'ineq',
                     'fun': lambda x: ca[typ-1, tx] - C_0},
                    {'type': 'ineq',
                     'fun': lambda x: (1 - alpha - beta) - gammau[0, tx]},
                    {'type': 'ineq',
                     'fun': lambda x: (1 - alpha - beta) - gammas[0, tx]})                 
            res = minimize(util, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'xtol': 1e-8, 'disp': True})
            result[typ-1, tx] = res.fun
            nopt1[(typ-1)*2:(typ-1)*2+2, tx] = res.x
            print(i, tx, typ)
    for tx in range(1, Tmax):        
        wr[tx] = sum(wm[tx+20:tx+59])/sum(wa[tx+20:tx+59])
        er[tx] = wr[tx] - (ts/tu)
        nopt[:, tx] = stepsize * nopt1[:, tx] + ( 1- stepsize) * nopt2[:, tx]
        nopt2[:, tx] = nopt[:, tx]