"""

#"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# == Define parameters == #
eta_a = 0.01                        # Coefficient of technological growth in Agriculture
eta_m = 0.01                        # Coefficient of technological growth in Manufacturing
delta = 0.9                         # Power of Technological growth function
alpha = 0.002182                    # Coefficient of Agricultural consumption in Utility function
beta = 0.597818                     # Coefficient of Manufacturing consumption in Utility function
Aa_0 = 17.71453                     # Initial level of technology in Agriculture (10 @ 1900)
Am_0 = 51.7524                      # Initial level of technology in Manufacturing (45 @ 1900)
C_0 = 2.43                          # Minimum level of Agricultural consumption
eps = 0.6                           # Power of Manufacturing input in production function
tu = 0.20                           # Time spent to raise an unskilled child
ts = 0.40                           # Time spent to raise a skilled child
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
tol = 0.05                          # Error margin
stepsize = 0.1                      # Step size
tcnt1 = [0]*T                       # Number of points within the error band (for the whole time period)
tcnt2 = [0]*T                       # Number of points within the error band (for the 1900-2010 period)
niter = 5                          # number of iterations

PopData =  [0]*Tmax
GDPData =  [0]*Tmax
PopData[0:161] = [19.98519, 21.1073662, 22.2295424, 23.3517186, 24.4738948, 25.596071, 26.7182472, 27.8404234, 28.9625996, 30.0847758, 31.206952, 32.3291282, 33.4513044, 34.5734806, 35.6956568, 36.817833, 37.9400092, 39.0621854, 40.1843616, 41.3065378, 42.428714, 43.5508902, 44.6730664, 45.7952426, 46.9174188, 48.039595, 49.1617712, 50.2839474, 51.4061236, 52.5282998, 53.650476, 54.7726522, 55.8948284, 57.0170046, 58.1391808, 59.261357, 60.3835332, 61.5057094, 62.6278856, 63.7500618, 64.872238, 65.9944142, 67.1165904, 68.2387666, 69.3609428, 70.483119, 71.6052952, 72.7274714, 73.8496476, 74.9718238, 76.094, 77.584, 79.163, 80.632, 82.166, 83.822, 85.45, 87.008, 88.71, 90.49, 92.407, 93.863, 95.335, 97.225, 99.111, 100.546, 101.961, 103.268, 103.208, 104.514, 106.461, 108.538, 110.049, 111.947, 114.109, 115.829, 117.397, 119.035, 120.509, 121.767, 123.076741, 124.039648, 124.840471, 125.578763, 126.373773, 127.250232, 128.05318, 128.824829, 129.824939, 130.879718, 132.122446, 133.402471, 134.859553, 136.739353, 138.397345, 139.928165, 141.388566, 144.126071, 146.631302, 149.18813, 152.271417, 154.877889, 157.55274, 160.184192, 163.025854, 165.931202, 168.903031, 171.98413, 174.881904, 177.829628, 180.671158, 183.691481, 186.537737, 189.241798, 191.888791, 194.302963, 196.560338, 198.712056, 200.706052, 202.676946, 205.052174, 207.660677, 209.896021, 211.908788, 213.853928, 215.973199, 218.035164, 220.239425, 222.584545, 225.055487, 227.224681, 229.465714, 231.664458, 233.791994, 235.824902, 237.923795, 240.132887, 242.288918, 244.498982, 246.81923, 249.464396, 252.153092, 255.029699, 257.782608, 260.327021, 262.803276, 265.228572, 267.783607, 270.248003, 272.690813, 282.162411, 284.968955, 287.625193, 290.107933, 292.805298, 295.516599, 298.379912, 301.231207, 304.093966, 306.771529, 309.326295]
GDPData[0:161] = [36.94592673, 40.60647561, 44.93905721, 50.42852705, 53.00303949, 54.22138757, 57.78112148, 58.88973673, 61.56155015, 65.6809459, 69.92971089, 70.94685437, 75.71644891, 83.3098721, 88.84308378, 86.89888999, 88.84454829, 94.48793332, 98.48462132, 103.9134939, 103.7230869, 109.0012085, 113.5115306, 119.2623777, 118.5722833, 124.834994, 126.3631812, 130.506633, 136.0186345, 152.8275793, 170.8206998, 176.1186934, 186.5746962, 190.3562319, 193.0383942, 193.7822098, 198.9072868, 207.1668892, 205.523659, 217.5968392, 220.0400326, 228.8200335, 250.2103527, 237.3625434, 229.8482631, 256.8560169, 250.9356066, 274.1449887, 279.1289038, 303.7383775, 311.2843682, 346.3243616, 349.9503442, 366.9477152, 362.3136083, 389.1154617, 434.0111662, 440.6859863, 404.5722425, 454.0332119, 458.6839251, 473.6030283, 495.808589, 515.3634221, 475.6536238, 489.0751109, 556.5737868, 541.9232915, 584.0524613, 593.6820042, 591.1063235, 577.7188585, 609.654257, 690.0635894, 711.1901247, 727.6862954, 775.1069086, 782.833552, 791.6652504, 840.0367008, 764.6404327, 705.9551092, 612.7626958, 599.8791367, 646.2258852, 695.656397, 794.4270467, 828.3592998, 795.3680321, 858.6694579, 926.1304141, 1094.658402, 1313.681091, 1574.987328, 1706.916688, 1638.369567, 1300.286023, 1280.703457, 1329.148565, 1334.30049, 1455.919987, 1566.782877, 1625.242318, 1699.972038, 1688.802488, 1808.128201, 1843.455338, 1878.06442, 1859.086979, 1997.056822, 2046.72879, 2094.401484, 2220.728869, 2316.762527, 2450.912331, 2607.293504, 2778.090777, 2847.549802, 2983.081773, 3076.51618, 3081.902615, 3178.101057, 3346.554335, 3536.618462, 3526.722813, 3516.82824, 3701.165784, 3868.836466, 4089.53964, 4228.65615, 4221.236212, 4326.703361, 4245.279054, 4423.381089, 4745.425997, 4929.144101, 5099.482511, 5278.916726, 5501.090572, 5691.473288, 5787.713765, 5757.358609, 5938.36192, 6094.009874, 6329.275922, 6474.770998, 6700.496375, 6982.253882, 7267.901388, 7599.8417, 8098.606988, 8186.045079, 8334.495139, 8546.267117, 8842.661691, 9114.218185, 9356.459903, 9535.451794, 9503.32157, 9172.052801, 9431.774587]


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

# == Initializing @ yesr 1850 == #
u[0:20, 0] = [0.521743004, 0.576141049, 0.604033241, 0.577785019, 0.592997873, 0.574931298, 0.566337374, 0.53020305, 0.568835914, 0.468570899, 0.542695061, 0.432128187, 0.520871421, 0.443058421, 0.482480313, 0.415404037, 0.453997221, 0.424748101, 0.460572766, 0.386064659]
s[0:20, 0] = [0.010647816, 0.011757981, 0.012327209, 0.011791531, 0.012101997, 0.011733292, 0.011557906, 0.01082047, 0.011608896, 0.009562671, 0.011075409, 0.008818943, 0.010630029, 0.009042009, 0.009846537, 0.008477633, 0.009265249, 0.008668329, 0.009399444, 0.007878871]
lg[0:20, 0] = [0.446421811, 0.371810834, 0.419369842, 0.383888256, 0.362532674, 0.430472223, 0.345244229, 0.285946899, 0.343426378, 0.225143132, 0.473120206, 0.189385137, 0.254521513, 0.219127334, 0.20774185, 0.312474431, 0.202371421, 0.188078023, 0.212766477, 0.15605274]
mg[0:20, 0] = [0.009110649, 0.007587976, 0.008558568, 0.007834454, 0.007398626, 0.008785147, 0.007045801, 0.005835651, 0.007008702, 0.004594758, 0.009655514, 0.003865003, 0.005194317, 0.004471986, 0.00423963, 0.006377029, 0.004130029, 0.003838327, 0.004342173, 0.00318475]
l[20:40, 0] = [0.33266442, 0.120067581, 0.159124148, 0.133149454, 0.140462195, 0.225396227, 0.130785263, 0.114789164, 0.13320014, 0.10811506, 0.243859349, 0.079458743, 0.110726417, 0.077495862, 0.083899691, 0.112028092, 0.083800525, 0.061392335, 0.068472894, 0.050025207]
m[20:40, 0] = [0.00678907, 0.002450359, 0.003247432, 0.002717336, 0.002866575, 0.004599923, 0.002669087, 0.002342636, 0.00271837, 0.00220643, 0.004976721, 0.001621607, 0.002259723, 0.001581548, 0.001712239, 0.002286288, 0.001710215, 0.001252905, 0.001397406, 0.001020923]
for i in range(20):
    gammau[i, 0] = 0.62586
    gammas[i, 0] = 0
    l[i, 0] = lg[i, 0] * (1 - gammau[i, 0])
    m[i, 0] = mg[i, 0] * (1 - gammas[i, 0])
ct[0] = sum(u[:, 0] + s[:, 0])
mt[0] = sum(m[:,0])
lt[0] = sum(l[:,0])
B[0] = (mt[0]/(mt[0] + lt[0]))**delta
Pop[0] = ct[0] + sum(lg[:, 0]) + sum(mg[:, 0]) + sum(l[20:40, 0]) + sum(m[20:40, 0])
lmt[0] = (((beta/alpha)/Aa[0]) * (1 - eps) * (Aa[0] * lt[0] - Pop[0] * C_0))/(1 + (beta/alpha) * (1 - eps))
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
for i in range(niter):
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
    x = range(1850, 1850 + Tmax)
    
    plt.plot(x[0:Tmax], Pop[0:Tmax], 'b', label ="Model")
    plt.plot(x[0:Tmax], PopData[0:Tmax], 'r', label ="Data")
    plt.xlabel('Time')
    plt.ylabel('Population (millions)')
    plt.title('Population Growth')
    axes = plt.gca()
    axes.set_xlim([1850,2010])
    plt.legend(loc=1, prop={'size':8})
    plt.xticks(np.arange(min(x), 2010, 25))
    plt.show()
    
    plt.plot(x[0:Tmax], Y[0:Tmax], 'b', label ="Model")
    plt.plot(x[0:Tmax], GDPData[0:Tmax], 'r', label ="Data")
    plt.xlabel('Time')
    plt.ylabel('GDP')
    plt.title('Output Growth')
    axes = plt.gca()
    axes.set_xlim([1850,2010])
    plt.legend(loc=1, prop={'size':8})
    plt.xticks(np.arange(min(x), 2010, 25))    
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
    
    plt.plot(x[0:Tmax], nopt2[0, 0:Tmax], 'm--', label = "unskilled children: previous run")
    plt.plot(x[0:Tmax], nopt1[0, 0:Tmax], 'm', label = "unskilled children: current run")
    plt.plot(x[0:Tmax], nopt2[1, 0:Tmax], 'c--', label = "skilled children: previous run")
    plt.plot(x[0:Tmax], nopt1[1, 0:Tmax], 'c', label = "skilled children: current run")  
    plt.xlabel('Time')
    plt.ylabel('Number of children')
    plt.title('Number of children of the unskilled parents')
    axes = plt.gca()
    axes.set_xlim([1900,2010])
    plt.legend(loc=1, prop={'size':8})
    plt.show()
    
    plt.plot(x[0:Tmax], nopt2[2, 0:Tmax], 'm--', label = "unskilled children: previous run")
    plt.plot(x[0:Tmax], nopt1[2, 0:Tmax], 'm', label = "unskilled children: current run")
    plt.plot(x[0:Tmax], nopt2[3, 0:Tmax], 'c--', label = "skilled children: previous run")
    plt.plot(x[0:Tmax], nopt1[3, 0:Tmax], 'c', label = "skilled children: current run")  
    plt.xlabel('Time')
    plt.ylabel('Number of children')
    plt.title('Number of children of the skilled parents')
    axes = plt.gca()
    axes.set_xlim([1900,2010])
    plt.legend(loc=1, prop={'size':8})
    plt.show()
 
    plt.plot(x[0:Tmax], gammau[0, 0:Tmax], 'm', label = "unskilled parents")
    plt.plot(x[0:Tmax], gammas[0, 0:Tmax], 'c', label = "skilled parents")  
    plt.xlabel('Time')
    plt.ylabel('Ratio')
    plt.title('Parenting time as a portion of the total time')
    axes = plt.gca()
    axes.set_xlim([1900,2010])
    plt.legend(loc=1, prop={'size':8})
    plt.show()
   
    print(i, round(tcnt1[i]*100/T, 2),'%', round(tcnt2[i]/1.11, 2),'%')