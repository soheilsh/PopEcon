"""

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# == Define parameters == #
eta_a = 0.01                        # Coefficient of technological growth in Agriculture
eta_m = 0.02                        # Coefficient of technological growth in Manufacturing
delta = 0.1                         # Power of Technological growth function
alpha = 0.1                         # Coefficient of Agricultural consumption in Utility function
beta = 0.7                          # Coefficient of Manufacturing consumption in Utility function
Aa_0 = 10                           # Initial level of technology in Agriculture
Am_0 = 45                           # Initial level of technology in Manufacturing
C_0 = 0.01                          # Minimum level of Agricultural consumption
eps = 0.6                           # Power of Manufacturing input in production function
tu = 0.1                            # Time spent to raise an unskilled child
ts = 0.3                            # Time spent to raise a skilled child
T = 200                             # Time horizon
Tmax = 1                           # Max Modeling Time
stepsize = 0.1                      # Step size

# == Generate age matrix == #
u = np.zeros((20, T))               # number of Unskilled children
s = np.zeros((20, T))               # number of Skilled children
l = np.zeros((40, T))               # Effective number of Unskilled adults
lg = np.zeros((20, T))              # Gross number of Unskilled parents
m = np.zeros((40, T))               # Effectivenumber of Skilled adults
mg = np.zeros((20, T))              # Gross number of Skilled parents

# == Labor == #
ct = [0]*T                          # Total number of children
mt = [0]*T                          # Total number of skilled adults in Manufacturing
lt = [0]*T                          # Total number of unskilled adults
lmt = [0]*T                         # Total number of unskilled adults in Manufacturing 
lat = [0]*T                         # Total number of unskilled adults in Agriculture 
Pop = [0]*T                         # Total Population

# == Generate education matrix == #
gammau = np.zeros((20, T))          # Child rearing time spent by unskilled parents
gammas = np.zeros((20, T))          # Child rearing time spent by skilled parents

# == Prices == #
pa = [0]*T                          # Pice of Agricultural good

# == Wages == #
wu = [0]*T                          # Wage of unskilled labor
ws = [0]*T                          # Wage of skilled labor

# == Technology == #
B = [0]*T                           # Technological growth underlying function
Aa = [Aa_0]*T                       # Level of technology in Agricultural sector
Am = [Am_0]*T                       # Level of technology in Manufacturing sector

# == Output == #
Y = [0]*T                           # Total output
Ya = [0]*T                          # Agricultural output
Ym = [0]*T                          # Manufacturing output

# == Consumption == #
ca = np.zeros((2, T))               # Consumption of Agricultural good
cm = np.zeros((2, T))               # Consumption of Manufacturing good

# == Utility == #
W = np.zeros((2, T))                # Utility

# == Result == #
result = np.zeros((2, T))
nopt = 0.5 + np.zeros((4, T))      # Optimal number of children (updated)
nopt1 = np.zeros((4, T))            # Optimal number of children (new)
nopt2 = np.zeros((4, T))            # Optimal number of children (old)

# == Error == #
wr = [0]*T                          # Ratio of future wages
er = [0]*T                          # Difference between ts/tu and wr
tol = 0.1                           # Error margin

# Initializing
for i in range(20):
    u[i, 0] = 1.5
    s[i, 0] = 0.25
    l[i, 0] = 1
    m[i, 0] = 0.07
    l[i+20, 0] = 0.6
    m[i+20, 0] = 0.03
    gammau[i, 0] = 1-alpha-beta
    gammas[i, 0] = 1-alpha-beta
    lg[i, 0] = l[i, 0]/(1 - gammau[i, 0])
    mg[i, 0] = m[i, 0]/(1 - gammas[i, 0])
u[0, 0] = nopt[0, 0] * lg[0, 0] + nopt[2, 0] * mg[0, 0]
s[0, 0] = nopt[1, 0] * lg[0, 0] + nopt[3, 0] * mg[0, 0]
ct[0] = sum(u[:, 0] + s[:, 0])
mt[0] = sum(m[:,0])
lt[0] = sum(l[:,0])
B[0] = (mt[0]/(mt[0] + lt[0]))**delta
lmt[0] = (((beta/alpha)/Aa[0]) * (1 - eps) * (Aa[0] * lt[0] - (mt[0] + lt[0]) * C_0))/(1 + (beta/alpha) * (1 - eps))
lat[0] = lt[0] - lmt[0]
Ya[0] = Aa[0] * lat[0]
Ym[0] = Am[0] * mt[0]**eps * lmt[0]**(1 - eps)
pa[0] = (Am[0]/Aa[0]) * (1-eps) * (mt[0]/lmt[0])**eps
wu[0] = pa[0] * Aa[0]
ws[0] = eps * Am[0] * (mt[0]/lmt[0])**(eps - 1)
Pop[0] = ct[0] + sum(lg[:, 0]) + sum(mg[:, 0]) + sum(l[20:40, 0]) + sum(m[20:40, 0])
Y[0] = pa[0] * Ya[0] + Ym[0]
    
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
        lg[0, t] = u[19, t-1]
        mg[0, t] = s[19, t-1] 
        l[0, t] = u[19, t-1] * (1 - gammau[0, t])
        m[0, t] = s[19, t-1] * (1 - gammas[0, t])
        lg[1:20, t] = lg[0:19, t-1]
        mg[1:20, t] = mg[0:19, t-1]        
        l[1:20, t] = l[0:19, t-1]
        m[1:20, t] = m[0:19, t-1]
        l[20, t] = lg[19, t-1]
        m[20, t] = mg[19, t-1]         
        l[21:40, t] = l[20:39, t-1]
        m[21:40, t] = m[20:39, t-1]
        gammau[1:20, t] = gammau[0:19, t-1]
        gammas[1:20, t] = gammas[0:19, t-1]
        ct[t] = sum(u[:, t] + s[:, t])
        mt[t] = sum(m[:,t])
        lt[t] = sum(l[:,t])
        B[t] = (mt[t]/(mt[t] + lt[t]))**delta
        Aa[t] = (1 + eta_a * B[t-1]) * Aa[t-1]
        Am[t] = (1 + eta_m * B[t-1]) * Am[t-1]
        lmt[t] = (((beta/alpha)/Aa[t]) * (1 - eps) * (Aa[t] * lt[t] - (mt[t] + lt[t]) * C_0))/(1 + (beta/alpha) * (1 - eps))
        lat[t] = lt[t] - lmt[t]
        Ya[t] = Aa[t] * lat[t]
        Ym[t] = Am[t] * mt[t]**eps * lmt[t]**(1 - eps)
        pa[t] = (Am[t]/Aa[t]) * (1-eps) * (mt[t]/lmt[t])**eps
        wu[t] = pa[t] * Aa[t]
        ws[t] = eps * Am[t] * (mt[t]/lmt[t])**(eps - 1)
    if typ == 1:        #Unskilled
        ca[0, tx] = (wu[tx] * (1-gammau[0, tx]) + (beta/alpha) * pa[tx] * C_0)/((1+ (beta/alpha)) * pa[tx])
        cm[0, tx]  = (beta/alpha) * pa[tx] * (ca[0, tx] - C_0)
        W[0, tx]  = -(alpha * np.log(ca[0, tx] - C_0) + beta * np.log(cm[0, tx] ) + (1 - alpha - beta) * np.log(n[0] * np.mean(wu[tx+20:tx+60]) + n[1] * np.mean(ws[tx+20:tx+60])))
    else:              #Skilled
        ca[1, tx] = (ws[tx] * (1-gammas[0, tx]) + (beta/alpha) * pa[tx] * C_0)/((1+ (beta/alpha)) * pa[tx])
        cm[1, tx]  = (beta/alpha) * pa[tx] * (ca[1, tx] - C_0)
        W[1, tx]  = -(alpha * np.log(ca[1, tx] - C_0) + beta * np.log(cm[1, tx] ) + (1 - alpha - beta) * np.log(n[0] * np.mean(wu[tx+20:tx+60]) + n[1] * np.mean(ws[tx+20:tx+60])))
    return W[typ-1, tx]

for i in range(1):
    for tx in range(1,Tmax + 40):
        nopt2[:, tx] = nopt[:, tx]
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
            res = minimize(util, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'xtol': 1e-8, 'disp': False})
            result[typ-1, tx] = res.fun
            nopt1[(typ-1)*2:(typ-1)*2+2, tx] = res.x
        print(i, tx)
        nuu = nopt1[0, tx]
        nsu = nopt1[1, tx]
        nus = nopt1[2, tx]
        nss = nopt1[3, tx]
        gammau[0, tx] = nuu * tu + nsu * ts
        gammas[0, tx] = nus * tu + nss * ts       
        uu = nuu * u[19, tx-1] 
        su = nsu * u[19, tx-1]
        us = nus * s[19, tx-1]
        ss = nss * s[19, tx-1]
        u[0, tx] = uu + us
        s[0, tx] = su + ss
        u[1:20, tx] = u[0:19, tx-1]
        s[1:20, tx] = s[0:19, tx-1]
        lg[0, tx] = u[19, tx-1]
        mg[0, tx] = s[19, tx-1]
        l[0, tx] = u[19, tx-1] * (1 - gammau[0, tx])
        m[0, tx] = s[19, tx-1] * (1 - gammas[0, tx])             
        l[1:20, tx] = l[0:19, tx-1]
        m[1:20, tx] = m[0:19, tx-1]
        lg[1:20, tx] = lg[0:19, tx-1]
        mg[1:20, tx] = mg[0:19, tx-1]        
        l[20, tx] = lg[19, tx-1]
        m[20, tx] = mg[19, tx-1]
        l[21:40, tx] = l[20:39, tx-1]
        m[21:40, tx] = m[20:39, tx-1]
        gammau[1:20, tx] = gammau[0:19, tx-1]
        gammas[1:20, tx] = gammas[0:19, tx-1]
        ct[tx] = sum(u[:, tx] + s[:, tx])
        mt[tx] = sum(m[:,tx])
        lt[tx] = sum(l[:,tx])
        B[tx] = (mt[tx]/(mt[tx] + lt[tx]))**delta
        Aa[tx] = (1 + eta_a * B[tx-1]) * Aa[tx-1]
        Am[tx] = (1 + eta_m * B[tx-1]) * Am[tx-1]
        lmt[tx] = (((beta/alpha)/Aa[tx]) * (1 - eps) * (Aa[tx] * lt[tx] - (mt[tx] + lt[tx]) * C_0))/(1 + (beta/alpha) * (1 - eps))
        lat[tx] = lt[tx] - lmt[tx]
        Ya[tx] = Aa[tx] * lat[tx]
        Ym[tx] = Am[tx] * mt[tx]**eps * lmt[tx]**(1 - eps)
        pa[tx] = (Am[tx]/Aa[tx]) * (1-eps) * (mt[tx]/lmt[tx])**eps
        wu[tx] = pa[tx] * Aa[tx]
        ws[tx] = eps * Am[tx] * (mt[tx]/lmt[tx])**(eps - 1)    
        wr[tx] = sum(ws[tx+20:tx+60])/sum(wu[tx+20:tx+60])
        er[tx] = wr[tx] - (ts/tu)
        nopt[:, tx] = stepsize * nopt1[:, tx] + ( 1- stepsize) * nopt2[:, tx]
        Pop[tx] = ct[tx] + sum(lg[:, tx]) + sum(mg[:, tx]) + sum(l[20:40, tx]) + sum(m[20:40, tx])
        Y[tx] = pa[tx] * Ya[tx] + Ym[tx]
    x = range(T)
    plt.plot(x[0:Tmax], Pop[0:Tmax])
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Population Growth')
    plt.show()
    plt.plot(x[0:Tmax], Y[0:Tmax])
    plt.xlabel('Time')
    plt.ylabel('GDP')
    plt.title('Output Growth')    
    plt.show()
    plt.plot(x[0:Tmax], wr[0:Tmax], 'r')
    plt.plot(x[0:Tmax], [ts/tu]*Tmax, 'g')
    plt.xlabel('Time')
    plt.ylabel('Ratio')
    plt.title('Wages ratio')
    plt.show()    