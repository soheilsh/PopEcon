"""

#"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# == Define parameters == #
eta_a = 0.01                        # Coefficient of technological growth in Agriculture
eta_m = 0.01                        # Coefficient of technological growth in Manufacturing
delta = 0.9                         # Power of Technological growth function
alpha = 0.002                       # Coefficient of Agricultural consumption in Utility function
beta = 0.598                        # Coefficient of Manufacturing consumption in Utility function
Aa_0 = 50                           # Initial level of technology in Agriculture (10 @ 1900)
Am_0 = 100                          # Initial level of technology in Manufacturing (45 @ 1900)
C_0 = 2.43                          # Minimum level of Agricultural consumption
eps = 0.6                           # Power of Manufacturing input in production function
tu = 0.05                            # Time spent to raise an unskilled child
ts = 0.1                            # Time spent to raise a skilled child
ro = 0.015                          # Annual social time preference discount rate 
Tmax = 200                          # Max Modeling Time
T = Tmax + 101                      # Time horizon

# == Age matrix == #
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

# == Education matrix == #
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
nopt = 0.5 + np.zeros((4, T))       # Optimal number of children (updated)
nopt1 = 0.5 + np.zeros((4, T))      # Optimal number of children (new)
nopt2 = 0.5 + np.zeros((4, T))      # Optimal number of children (old)

# == Error == #
wr = [0]*T                          # Ratio of future wages
er = [0]*T                          # Difference between ts/tu and wr
tol = 0.1                           # Error margin
stepsize = 0.1                      # Step size
tcnt1 = [0]*T                       # Number of points within the error band (for the whole time period)
tcnt2 = [0]*T                       # Number of points within the error band (for the 1900-2010 period)

PopData =  [0]*Tmax
GDPData =  [0]*Tmax
PopData[0:171] = [17.063,17.676,18.289,18.902,19.515,20.128,20.740,21.353,21.966,22.579,23.192,24.017,24.842,25.667,26.492,27.318,28.143,28.968,29.793,30.618,31.443,32.155,32.866,33.578,34.289,35.001,35.712,36.424,37.135,37.847,38.558,39.721,40.885,42.048,43.211,44.374,45.537,46.700,47.863,49.026,50.189,51.468,52.747,54.026,55.305,56.584,57.864,59.143,60.422,61.701,62.980,64.291,65.602,66.913,68.224,69.535,70.846,72.157,73.468,74.779, 76.090 , 77.580 , 79.160 , 80.630 , 82.170 , 83.820 , 85.450 , 87.010 , 88.710 , 90.490 , 92.410 , 93.860 , 95.340 , 97.230 , 99.110 , 100.550 , 101.960 , 103.270 , 103.210 , 104.510 , 106.460 , 108.540 , 110.050 , 111.950 , 114.110 , 115.830 , 117.400 , 119.040 , 120.510 , 121.770 , 123.080 , 124.040 , 124.840 , 125.580 , 126.370 , 127.250 , 128.050 , 128.820 , 129.820 , 130.880 , 132.120 , 133.400 , 134.860 , 136.740 , 138.400 , 139.930 , 141.390 , 144.130 , 146.630 , 149.190 , 152.270 , 154.880 , 157.550 , 160.180 , 163.030 , 165.930 , 168.900 , 171.980 , 174.880 , 177.830 , 180.670 , 183.690 , 186.540 , 189.240 , 191.890 , 194.300 , 196.560 , 198.710 , 200.710 , 202.680 , 205.050 , 207.660 , 209.900 , 211.910 , 213.850 , 215.970 , 218.040 , 220.240 , 222.580 , 225.060 , 227.220 , 229.470 , 231.660 , 233.790 , 235.820 , 237.920 , 240.130 , 242.290 , 244.500 , 246.820 , 249.620 , 252.980 , 256.510 , 259.920 , 263.130 , 266.280 , 269.390 , 272.650 , 275.850 , 279.040 , 282.160,  284.970,287.630,290.110,292.810,295.520,298.380,301.230,304.090,306.770,309.330]
GDPData[60:171] = [442.757,492.587,497.777,521.930,515.419,553.429,617.304,626.841,575.495,645.734,652.469,673.654,705.295,733.106,676.514,695.640,791.684,770.862,830.750,844.335,840.707,821.779,867.179,981.512,1011.649,1034.970,1102.434,1113.432,1125.981,1194.910,1087.671,1004.059,871.500,853.266,919.207,989.500,1129.953,1178.157,1131.169,1221.384,1317.333,1557.027,1868.510,2240.173,2427.804,2330.448,1849.583,1821.669,1890.599,1897.931,2070.744,2228.502,2311.738,2417.994,2402.117,2571.821,2621.938,2671.218,2644.375,2840.492,2911.040,2979.033,3158.713,3295.142,3486.214,3708.530,3951.564,4050.184,4243.110,4375.855,4383.566,4520.294,4760.127,5030.262,5016.087,5002.226,5264.472,5503.039,5816.678,6014.656,6003.866,6154.380,6038.150,6291.525,6749.665,7010.785,7253.172,7508.647,7824.398,8095.238,8237.490,8198.682,8453.186,8679.428,9031.014,9273.203,9604.513,10012.759,10444.161,10872.930,11289.080,11328.143,11874.496,12207.767,12669.913,13093.700,13442.885,13681.961,13642.130,13263.417,13599.264]


def state(nuu, nsu, nus, nss, t):
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
    Pop[t] = ct[t] + sum(lg[:, t]) + sum(mg[:, t]) + sum(l[20:40, t]) + sum(m[20:40, t])
    lmt[t] = (((beta/alpha)/Aa[t]) * (1 - eps) * (Aa[t] * lt[t] - Pop[t] * C_0))/(1 + (beta/alpha) * (1 - eps))
    lat[t] = lt[t] - lmt[t]
    Ya[t] = Aa[t] * lat[t]
    Ym[t] = Am[t] * mt[t]**eps * lmt[t]**(1 - eps)
    pa[t] = (Am[t]/Aa[t]) * (1-eps) * (mt[t]/lmt[t])**eps
    Y[t] = pa[t] * Ya[t] + Ym[t]
    wu[t] = pa[t] * Aa[t]
    ws[t] = eps * Am[t] * (mt[t]/lmt[t])**(eps - 1)

def util(n):
    for t in range(tx, tx + 60):
        if t == tx:
            if typ == 1:                    #Unskilled
                nuu = n[0]
                nsu = n[1]
                nus = nopt[2, tx]
                nss = nopt[3, tx]
            else:                           #Skilled
                nuu = nopt[0, tx]
                nsu = nopt[1, tx]
                nus = n[0]
                nss = n[1]
        else:
            nuu = nopt[0, t]
            nsu = nopt[1, t]
            nus = nopt[2, t]
            nss = nopt[3, t]
        state(nuu, nsu, nus, nss, t)
    if typ == 1:                            #Unskilled
        ca[0, tx] = (wu[tx] * (1-gammau[0, tx]) + (beta/alpha) * pa[tx] * C_0)/((1+ (beta/alpha)) * pa[tx])
        cm[0, tx]  = (beta/alpha) * pa[tx] * (ca[0, tx] - C_0)
        W[0, tx]  = -(alpha * np.log(ca[0, tx] - C_0) + beta * np.log(cm[0, tx] ) + (1 - alpha - beta) * np.log(n[0] * np.mean(wu[tx+20:tx+60]) + n[1] * np.mean(ws[tx+20:tx+60])))
    else:                                   #Skilled
        ca[1, tx] = (ws[tx] * (1-gammas[0, tx]) + (beta/alpha) * pa[tx] * C_0)/((1+ (beta/alpha)) * pa[tx])
        cm[1, tx]  = (beta/alpha) * pa[tx] * (ca[1, tx] - C_0)
        W[1, tx]  = -(alpha * np.log(ca[1, tx] - C_0) + beta * np.log(cm[1, tx] ) + (1 - alpha - beta) * np.log(n[0] * np.mean(wu[tx+20:tx+60]) + n[1] * np.mean(ws[tx+20:tx+60])))
    return W[typ-1, tx]

# == Initializing @ yesr 1900 == #
#    u[i, 0] = 1.5
#    s[i, 0] = 0.25
#    l[i, 0] = 1
#    m[i, 0] = 0.07
#    l[i+20, 0] = 0.6
#    m[i+20, 0] = 0.03

# == Initializing @ yesr 1840 == #
for i in range(20):
    u[i, 0] = 0.45
    s[i, 0] = 0.02
    l[i, 0] = 0.25
    m[i, 0] = 0.015
    l[i+20, 0] = 0.12
    m[i+20, 0] = 0.006
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
for t in range(1, T):
    state(nopt[0, t], nopt[1, t], nopt[2, t], nopt[3, t], t)
for i in range(20):
    tcnt1[i]=0
    tcnt2[i]=0
    wr[0] = sum(ws[20:60])/sum(wu[20:60])
    for tx in range(1,Tmax + 41):
        wr[tx] = sum(ws[tx+20:tx+60])/sum(wu[tx+20:tx+60])
        er[tx] = wr[tx] - (ts/tu)
        nratio = np.mean([nopt[0, tx]/nopt[1, tx], nopt[2, tx]/nopt[3, tx]])
        if er[tx] > tol:
            tst = 's'
            bnds = [(0, 0), (0, None)]
            cons = ({'type': 'eq',
                     'fun': lambda x: x[0]})        
        else:
            if er[tx] < -tol:
                tst = 'u'
                bnds = [(0, None), (0, 0)]
                cons = ({'type': 'eq',
                         'fun': lambda x: x[1]})
            else:
                tcnt1[i] = tcnt1[i] + 1
                if (tx > 59) & (tx < 171):
                    tcnt2[i] = tcnt2[i] + 1
                tst = 's&u'
                bnds = [(0, None), (0, None)]
                cons = ({'type': 'eq',
                         'fun': lambda x: x[0] / x[1] - nratio})
        x0 = [0.01, 0.01]
        for typ in [1, 2]:
            res = minimize(util, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'xtol': 1e-8, 'disp': False})
            result[typ-1, tx] = res.fun
            nopt1[(typ-1)*2:(typ-1)*2+2, tx] = res.x
#            print(i, tx, typ, tst , tcnt1[i], tcnt2[i],':  ', round(res.x[0], 3), round(res.x[1], 3), 'nratio:', round(nratio, 3))
        nuu = nopt1[0, tx]
        nsu = nopt1[1, tx]
        nus = nopt1[2, tx]
        nss = nopt1[3, tx]
        state(nuu, nsu, nus, nss, tx)
        for t in range(tx+1, tx+60):
            state(nopt[0, t], nopt[1, t], nopt[2, t], nopt[3, t], t)
    for t in range(1, T):
        nopt2[:, t] = nopt[:, t]
        nopt[:, t] = stepsize * nopt1[:, t] + ( 1- stepsize) * nopt2[:, t]
        state(nopt[0, t], nopt[1, t], nopt[2, t], nopt[3, t], t)
    x = range(1840, 1840 + Tmax)
    plt.plot(x[0:Tmax], Pop[0:Tmax], 'b', label ="Model")
    plt.plot(x[0:Tmax], PopData[0:Tmax], 'r', label ="Data")
    plt.xlabel('Time')
    plt.ylabel('Population (millions)')
    plt.title('Population Growth')
    axes = plt.gca()
    axes.set_xlim([1840,2010])
    plt.legend(loc=1, prop={'size':8})
    plt.show()
    plt.plot(x[0:Tmax], Y[0:Tmax], 'b', label ="Model")
    plt.plot(x[0:Tmax], GDPData[0:Tmax], 'r', label ="Data")
    plt.xlabel('Time')
    plt.ylabel('GDP')
    plt.title('Output Growth')
    axes = plt.gca()
    axes.set_xlim([1840,2010])
    plt.legend(loc=1, prop={'size':8})
    plt.show()
    plt.plot(x[0:Tmax], wr[0:Tmax], 'r')
    plt.plot(x[0:Tmax], [ts/tu]*Tmax, 'g')
    plt.plot(x[0:Tmax], [j + tol for j in [ts/tu]*Tmax] , 'g--')
    plt.plot(x[0:Tmax], [j - tol for j in [ts/tu]*Tmax], 'g--')    
    plt.xlabel('Time')
    plt.ylabel('Ratio')
    plt.title('Wages ratio')
    axes = plt.gca()
    axes.set_xlim([1900,2010])    
    plt.show()
    print(i, round(tcnt1[i]*100/T, 2),'%', round(tcnt2[i]/1.1, 2),'%')