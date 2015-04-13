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
T = 200            # Time horizon
tu = 0.1           # Time spent to raise an unskilled child
ts = 0.3           # Time spent to raise a skilled child

# == Generate age matrix == #
u = 0*np.empty((20, T))           # Unskilled children
s = 0*np.empty((20, T))           # Skilled children
l = 0*np.empty((40, T))           # Unskilled adults (Total)
m = 0*np.empty((40, T))           # Skilled adults
lm = 0*np.empty((40, T))          # Unskilled adults in Manufacturing 
la = 0*np.empty((40, T))          # Unskilled adults in Agriculture

# == Labor == #
mt = [0]*T          # Total skilled adults in Manufacturing
lt = [0]*T          # Total unskilled adults
lmt = [0]*T         # Total unskilled adults in Manufacturing 
lat = [0]*T         # Total unskilled adults in Agriculture 

# == Generate education matrix == #
gammau = 0*np.empty((20, T))
gammas = 0*np.empty((20, T))

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

# == Agricultural Consumption == #
ca = [0]*T

# == Result == #
result = [0]*T
nopt = 0*np.empty((4, T))


# Initializing
for i in range(19):
    u[i, 0] = 1.5
    s[i, 0] = 0.25
    l[i, 0] = 1
    m[i, 0] = 0.07
    l[i+20, 0] = 0.6
    m[i+20, 0] = 0.03    
    gammau[i+1, :] = 1-alpha-beta
    gammas[i+1, :] = 1-alpha-beta
u[0, :] = 1
s[0, :] = 0.3   
    
def util(n):
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
    uu = nuu * l[0, tx]
    su = nsu * l[0, tx]
    us = nus * m[0, tx]
    ss = nss * m[0, tx]
    gammau[1, tx] = nuu * tu + nus * ts
    gammas[1, tx] = nus * tu + nss * ts
    u[0, tx] = uu + us
    s[0, tx] = su + ss
    l[0, tx] = u[19, tx-1] * gammau[1, tx]
    m[0, tx] = s[19, tx-1] * gammas[1, tx]
    for t in range(tx, tx+59):     
        u[1:19, t] = u[0:18, t-1]
        s[1:19, t] = s[0:18, t-1]
        l[0, t] = u[19, t-1] * gammau[1, t]
        m[0, t] = s[19, t-1] * gammas[1, t]             
        l[1:39, t] = l[0:38, t-1]
        m[1:39, t] = m[0:38, t-1]    
        gammau[2:20, t] = gammau[1:19, t-1]
        gammas[2:20, t] = gammas[1:19, t-1]
        mt[t] = sum(m[:,t])
        lt[t] = sum(l[:,t])
        B[t] = mt[t]/(mt[t] + lt[t])
        Aa[t] = (1 + eta_a * B[t]) * Aa[t-1]
        Am[t] = (1 + eta_m * B[t]) * Am[t-1]
        lmt[t] = (((beta/alpha)/Aa[t]) * (1 - eps) * (Aa[t] * lt[t] - (mt[t] + lt[t]) * C_0))/(1 + (beta/alpha) * (1 - eps))
        lat[t] = lt[t] - lmt[t]
        Ya[t] = Aa[t] * lat[t]
        Ym[t] = Am[t] * mt[t]**eps * lmt[t]**(1 - eps)
        pa[t] = (Am[t]/Aa[t]) * (1-eps) * (mt[t]/lmt[t])**eps
        wa[t] = pa[t] * Aa[t]
        wm[t] = eps * Am[t] * (mt[t]/lmt[t])**(1 - eps)
    if typ == 1:        #Unskilled
        ca[tx] = (wa[tx] * (1-gammau[1, tx]) + (beta/alpha) * pa[tx] * C_0)/((1+ (beta/alpha)) * pa[tx])
        cm = (beta/alpha) * pa[tx] * (ca[tx] - C_0)
        w = alpha * np.log(ca[tx] - C_0) + beta * np.log(cm) + (1 - alpha - beta) * np.log(sum(nuu * wm[tx+20:tx+59]) + nsu * sum(wa[tx+20:tx+59]))
    else:              #Skilled
        ca[tx] = (wm[tx] * (1-gammas[1, tx]) + (beta/alpha) * pa[tx] * C_0)/((1+ (beta/alpha)) * pa[tx])
        cm = (beta/alpha) * pa[tx] * (ca[tx] - C_0)
        w = alpha * np.log(ca[tx] - C_0) + beta * np.log(cm) + (1 - alpha - beta) * np.log(sum(nuu * wm[tx+20:tx+59]) + nsu * sum(wa[tx+20:tx+59]))            
    return w

for tx in range(1,2):
    for typ in [1, 2]:
        x0 = [0, 0.7]
        cons = ({'type': 'ineq', ca[tx]:  C_0},
...         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6})
        bnds = tuple((0, None), (0, None))
        res = minimize(util, x0, method='SLSQP', bounds=bnds, options={'xtol': 1e-8, 'disp': True})
        result[tx] = res
        nopt[(typ-1)*2:(typ-1)*2+1, tx] = res.x
    wr = sum(wm[tx+20:tx+59])/sum(wa[tx+20:tx+59])
    er = abs(wr - (ts/tu))