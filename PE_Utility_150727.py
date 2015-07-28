import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import six 
from six.moves import zip
import csv
from numpy import genfromtxt
import xlsxwriter as xlwt

# =============================================== Define parameters =============================================== #

ro = 0.015                          # Annual social time preference discount rate
C_0 = 2.43                          # Minimum level of Agricultural consumption
eps = 0.6                           # Power of Manufacturing input in production function
delta = 0.9                         # Power of Technological growth function
B0 = 1                              # Coefficient of technological growth
eta_a = 0.02                        # Coefficient of technological growth in Agriculture
eta_m = 0.04                        # Coefficient of technological growth in Manufacturing
T = 561                             # Time horizon
Tmax = T - 59                       # Max Modeling Time
tol1 = 0.02                         # Error margin for constraints
tol2 = 0.02                         # Error margin for results
stepsize = 0.02                     # Step size
niter = 1                         # number of iterations
df = [0]*40                         # Discount factor
for t in range(40):
    df[t] = 1/(1 + ro)**t
    
# =============================================== Define parameters =============================================== #

# ================================================= Observed Data ================================================= #

h2000 = 1.0
PopData =  [0]*Tmax
GDPData =  [0]*Tmax
PopData[0:174] = [19.982, 20.278, 20.588, 20.912, 21.251, 21.607, 21.978, 22.367, 22.775, 23.201, 23.647, 24.114, 24.603, 25.115, 25.650, 26.211, 26.797, 27.411, 28.054, 28.727, 29.431, 30.168, 30.939, 31.747, 32.592, 33.476, 34.402, 35.371, 36.386, 37.447, 38.558, 39.721, 40.885, 42.048, 43.211, 44.374, 45.537, 46.700, 47.863, 49.026, 50.189, 51.468, 52.747, 54.026, 55.305, 56.584, 57.864, 59.143, 60.422, 61.701, 62.980, 64.291, 65.602, 66.913, 68.224, 69.535, 70.846, 72.157, 73.468, 74.779, 76.090, 77.580, 79.160, 80.630, 82.170, 83.820, 85.450, 87.010, 88.710, 90.490, 92.410, 93.860, 95.340, 97.230, 99.110, 100.550, 101.960, 103.270, 103.210, 104.510, 106.460, 108.540, 110.050, 111.950, 114.110, 115.830, 117.400, 119.040, 120.510, 121.770, 123.080, 124.040, 124.840, 125.580, 126.370, 127.250, 128.050, 128.820, 129.820, 130.880, 132.120, 133.400, 134.860, 136.740, 138.400, 139.930, 141.390, 144.130, 146.630, 149.190, 152.270, 154.880, 157.550, 160.180, 163.030, 165.930, 168.900, 171.980, 174.880, 177.830, 180.670, 183.690, 186.540, 189.240, 191.890, 194.300, 196.560, 198.710, 200.710, 202.680, 205.050, 207.660, 209.900, 211.910, 213.850, 215.970, 218.040, 220.240, 222.580, 225.060, 227.220, 229.470, 231.660, 233.790, 235.820, 237.920, 240.130, 242.290, 244.500, 246.820, 249.620, 252.980, 256.510, 259.920, 263.130, 266.280, 269.390, 272.650, 275.850, 279.040, 282.160, 284.970, 287.630, 290.110, 292.810, 295.520, 298.380, 301.230, 304.090, 306.770, 309.330, 311.590, 313.910, 316.160]
GDPData[0:174] = [108.002, 108.011, 108.021, 108.033, 108.049, 108.069, 108.094, 108.124, 108.163, 108.211, 108.271, 108.347, 108.441, 108.558, 108.705, 108.889, 109.119, 109.406, 109.765, 110.214, 110.775, 111.476, 112.353, 113.449, 114.818, 116.530, 118.670, 121.345, 124.689, 128.868, 134.093, 140.624, 146.777, 153.225, 159.860, 166.814, 173.971, 181.470, 189.258, 197.274, 205.595, 213.836, 222.301, 231.149, 240.398, 249.981, 259.912, 270.284, 281.028, 292.242, 303.855, 317.038, 347.857, 331.015, 321.586, 360.404, 353.092, 386.824, 395.001, 430.874, 442.757, 492.587, 497.777, 521.930, 515.419, 553.429, 617.304, 626.841, 575.495, 645.734, 652.469, 673.654, 705.295, 733.106, 676.514, 695.640, 791.684, 770.862, 830.750, 844.335, 840.707, 821.779, 867.179, 981.512, 1011.649, 1034.970, 1102.434, 1113.432, 1125.981, 1194.910, 1087.671, 1004.059, 871.500, 853.266, 919.207, 989.500, 1129.953, 1178.157, 1131.169, 1221.384, 1317.333, 1557.027, 1868.510, 2240.173, 2427.804, 2330.448, 1849.583, 1821.669, 1890.599, 1897.931, 2070.744, 2228.502, 2311.738, 2417.994, 2402.117, 2571.821, 2621.938, 2671.218, 2644.375, 2840.492, 2911.040, 2979.033, 3158.713, 3295.142, 3486.214, 3708.530, 3951.564, 4050.184, 4243.110, 4375.855, 4383.566, 4520.294, 4760.127, 5030.262, 5016.087, 5002.226, 5264.472, 5503.039, 5816.678, 6014.656, 6003.866, 6154.380, 6038.150, 6291.525, 6749.665, 7010.785, 7253.172, 7508.647, 7824.398, 8095.238, 8237.490, 8198.682, 8453.186, 8679.428, 9031.014, 9273.203, 9604.513, 10012.759, 10444.161, 10872.930, 11289.080, 11328.143, 11874.496, 12207.767, 12669.913, 13093.700, 13442.885, 13681.961, 13642.130, 13263.417, 13599.264, 13817.091, 14137.759, 14451.529]

# ================================================= Observed Data ================================================= #

# ============================================== Transition Function ============================================== #
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
    B[t] = B0 * (mt[t]/(mt[t] + lt[t]))**delta
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

# ============================================== Transition Function ============================================== #

# ================================================ Utility Function ================================================ #

def util(n):
    gamma = n[0] * tu + n[1] * ts 
    U[typ -1, tx] = 0
    wud = 0
    wsd = 0
    for t in range(tx + 20, tx + 60):
        wud = wud + wu[t] * df[t - (tx + 20)]
        wsd = wsd + ws[t] * df[t - (tx + 20)]
    for t in range(tx, tx + 20):
        if typ == 1:
            cax = (wu[t] * (1 - gamma) + (beta/alpha) * pa[t] * C_0)/((1 + (beta/alpha)) * pa[t])
            cmx = (beta/alpha) * pa[t] * (cax - C_0)
            W[t - tx] = alpha * np.log(cax - C_0) + beta * np.log(cmx) + (1 - alpha - beta) * np.log(n[0] * wud + n[1] * wsd)
        else:
            cax = (ws[t] * (1 - gamma) + (beta/alpha) * pa[t] * C_0)/((1 + (beta/alpha)) * pa[t])
            cmx = (beta/alpha) * pa[t] * (cax - C_0)
            W[t - tx] = alpha * np.log(cax - C_0) + beta * np.log(cmx) + (1 - alpha - beta) * np.log(n[0] * wud + n[1] * wsd)
        U[typ -1, tx] = U[typ -1, tx] + W[t - tx] * df[t - tx]
    return -U[typ -1, tx]

# ================================================ Utility Function ================================================ #

# =============================================== Export into Excel =============================================== #

def output(filename, sheet, list1, list2, list3, list4, val1, x, y, z, w):
    book = xlwt.Workbook(filename)
    sh = book.add_worksheet(sheet)

    v = [x, y, z, w]
    x_desc = 'alpha'
    y_desc = 'beta'
    z_desc = 'Tu'
    w_desc = 'Ts'
    desc = [x_desc, y_desc, z_desc, w_desc]

    row0_name = 'Time'
    
    row1_name = 'Nuu'
    row2_name = 'Nsu'
    row3_name = 'Nus'
    row4_name = 'Nss'
    
    row5_name = 'Population-Data'
    row6_name = 'Population'
    row7_name = 'GDP-Data'
    row8_name = 'GDP'
    
    row9_name = 'h'
    
    row10_name = 'Error'

    n = 0
    for v_desc, v_v in zip(desc, v):
        sh.write(n, 0, v_desc)
        sh.write(n, 1, v_v)
        n = n + 1
    
    sh.write(n, 0, row0_name)    
    sh.write(n + 1, 0, row1_name)
    sh.write(n + 2, 0, row2_name)
    sh.write(n + 3, 0, row3_name)
    sh.write(n + 4, 0, row4_name)
    sh.write(n + 5, 0, row5_name)
    sh.write(n + 6, 0, row6_name)
    sh.write(n + 7, 0, row7_name)
    sh.write(n + 8, 0, row8_name)
    sh.write(n + 9, 0, row9_name)
    sh.write(n + 10, 0, row10_name)
    
    for indx , m in enumerate(range(1840,2001), 1):
        sh.write(n + 0, indx, m)

    for indx , m in enumerate(list1[0,:], 1):
        sh.write(n + 1, indx, m)
        
    for indx , m in enumerate(list1[1,:], 1):
        sh.write(n + 2, indx, m)
    
    for indx , m in enumerate(list1[2,:], 1):
        sh.write(n + 3, indx, m)
    
    for indx , m in enumerate(list1[3,:], 1):
        sh.write(n + 4, indx, m)

    for indx , m in enumerate(PopData, 1):
        sh.write(n + 5, indx, m)
        
    for indx , m in enumerate(list2, 1):
        sh.write(n + 6, indx, m)

    for indx , m in enumerate(GDPData, 1):
        sh.write(n + 7, indx, m)

    for indx , m in enumerate(list3, 1):
        sh.write(n + 8, indx, m)

    for indx , m in enumerate(list4, 1):
        sh.write(n + 9, indx, m)
        
    sh.write(n + 10, 1, val1)    
    book.close()

# =============================================== Export into Excel =============================================== #

# ========================================== Calibrating the parameters =========================================== #
l1 = [0.2, 0.4, 0.6, 0.8]
l2 = [0.10, 0.15, 0.20, 0.25]
l3 = [1.0, 2.0, 3.0, 4.0]
for r1 in l1:
    for r2 in l2:
        for r3 in l3:
            alpha = r1 / 275                    # Coefficient of Agricultural consumption in Utility function
            beta = r1 - alpha                   # Coefficient of Manufacturing consumption in Utility function
            tu = r2                             # Time spent to raise a unskilled child
            ts = r3 * r2                        # Time spent to raise an skilled child
            
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
            h = [0]*T                           # Ratio of skilled to unskilled labor in Manufacturing
            
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
            Aa = [0]*T                          # Level of technology in Agricultural sector
            Am = [0]*T                          # Level of technology in Manufacturing sector
            
            # == Output == #
            Y = [0]*T                           # Total output
            Ya = [0]*T                          # Agricultural output
            Ym = [0]*T                          # Manufacturing output
            
            # == Utility == #
            W = [0]*20                          # Utility at each age
            U = np.zeros((2, Tmax))             # Welfare
                
            # == Result == #
            result = np.zeros((2, T))
            nopt = 0.5 + np.zeros((4, T))       # Optimal number of children (updated)
            nopt1 = 0.5 + np.zeros((4, T))      # Optimal number of children (new)
            nopt2 = 0.5 + np.zeros((4, T))      # Optimal number of children (old)
            
            # == Error == #
            wr = [0]*T                          # Ratio of future wages
            er1 = [0]*T                         # Difference between ts/tu and wr
            er2 = [0]*T                         # Difference between new and old solutions
            tcnt1 = [0]*niter                   # Number of points within the error band (for the whole time period)
            tcnt2 = [0]*niter                   # Number of points within the error band (for the 1900-2010 period)           
            
# ============================================ Initializing @ year 1840 ============================================ #

            u[0:20, 0] = [0.521743004, 0.576141049, 0.604033241, 0.577785019, 0.592997873, 0.574931298, 0.566337374, 0.53020305, 0.568835914, 0.468570899, 0.542695061, 0.432128187, 0.520871421, 0.443058421, 0.482480313, 0.415404037, 0.453997221, 0.424748101, 0.460572766, 0.386064659]
            s[0:20, 0] = [0.010647816, 0.011757981, 0.012327209, 0.011791531, 0.012101997, 0.011733292, 0.011557906, 0.01082047, 0.011608896, 0.009562671, 0.011075409, 0.008818943, 0.010630029, 0.009042009, 0.009846537, 0.008477633, 0.009265249, 0.008668329, 0.009399444, 0.007878871]
            lg[0:20, 0] = [0.446421811, 0.371810834, 0.419369842, 0.383888256, 0.362532674, 0.430472223, 0.345244229, 0.285946899, 0.343426378, 0.225143132, 0.473120206, 0.189385137, 0.254521513, 0.219127334, 0.20774185, 0.312474431, 0.202371421, 0.188078023, 0.212766477, 0.15605274]
            mg[0:20, 0] = [0.009110649, 0.007587976, 0.008558568, 0.007834454, 0.007398626, 0.008785147, 0.007045801, 0.005835651, 0.007008702, 0.004594758, 0.009655514, 0.003865003, 0.005194317, 0.004471986, 0.00423963, 0.006377029, 0.004130029, 0.003838327, 0.004342173, 0.00318475]
            l[20:40, 0] = [0.33266442, 0.120067581, 0.159124148, 0.133149454, 0.140462195, 0.225396227, 0.130785263, 0.114789164, 0.13320014, 0.10811506, 0.243859349, 0.079458743, 0.110726417, 0.077495862, 0.083899691, 0.112028092, 0.083800525, 0.061392335, 0.068472894, 0.050025207]
            m[20:40, 0] = [0.00678907, 0.002450359, 0.003247432, 0.002717336, 0.002866575, 0.004599923, 0.002669087, 0.002342636, 0.00271837, 0.00220643, 0.004976721, 0.001621607, 0.002259723, 0.001581548, 0.001712239, 0.002286288, 0.001710215, 0.001252905, 0.001397406, 0.001020923]
            
            for i in range(20):
                gammau[i, 0] = 1- alpha - beta 
                gammas[i, 0] = 1- alpha - beta
                l[i, 0] = lg[i, 0] * (1 - gammau[i, 0])
                m[i, 0] = mg[i, 0] * (1 - gammas[i, 0])
            ct[0] = sum(u[:, 0] + s[:, 0])
            mt[0] = sum(m[:,0])
            lt[0] = sum(l[:,0])
            B[0] = B0 * (mt[0]/(mt[0] + lt[0]))**delta
            Pop[0] = ct[0] + sum(lg[:, 0]) + sum(mg[:, 0]) + sum(l[20:40, 0]) + sum(m[20:40, 0])
            lat[0] = 0.7 * lt[0]
            lmt[0] = lt[0] - lat[0]
            h[0] = mt[0] / lmt[0]
            Y[0] = GDPData[0]
            Am[0] = Y[0] / ((1 - eps) * (h[0])**eps * lat[0] + (mt[0])**eps * (lmt[0])**(1 - eps))
            Ym[0] = Am[0] * mt[0]**eps * lmt[0]**(1 - eps)
            pa[0] = (beta/alpha * Y[0] - (1 + beta/alpha) * Ym[0]) / (beta/alpha * Pop[0] * C_0)
            Ya[0] = (Y[0] - Ym[0]) / pa[0] 
            Aa[0] = Ya[0] / lat[0]
            wu[0] = pa[0] * Aa[0]
            ws[0] = eps * Am[0] * (mt[0]/lmt[0])**(eps - 1)

# ============================================ Initializing @ year 1840 ============================================ #

# ================================== Testing the algorithm for a given set of n's ================================== #

#tst = genfromtxt('n.csv', delimiter=',')
#for t in range(1, T):
#    state(tst[0, t], tst[1, t], tst[2, t], tst[3, t], t)
#for tx in range(1, Tmax):
#    for typ in [1, 2]:
#        util(tst[(typ - 1) * 2:(typ - 1) * 2 + 2, tx])
#fl = open('u.csv', 'w')
#writer = csv.writer(fl)
#for values in U:
#    writer.writerow(values)
#fl.close()

# ================================== Testing the algorithm for a given set of n's ================================== #

# =========================== Optimization algorithm for finding the optimal set of n's =========================== #

            for t in range(1, T):
                state(nopt[0, t], nopt[1, t], nopt[2, t], nopt[3, t], t)
            for i in range(niter):
                tcnt1[i]=0
                tcnt2[i]=0
                wr[0] = sum([a * b for a,b in zip(ws[20:60], df)])/sum([a * b for a,b in zip(wu[20:60], df)])
                for tx in range(1, Tmax):
                    wr[tx] = sum([a * b for a,b in zip(ws[tx+20:tx+60], df)])/sum([a * b for a,b in zip(wu[tx+20:tx+60], df)])
                    er1[tx] = wr[tx] - (ts/tu)
                    nratio = np.mean([nopt[0, tx]/nopt[1, tx], nopt[2, tx]/nopt[3, tx]])
                    if er1[tx] > tol1:
                        tst = 's'
                        bnds = [(0, 0), (0, None)]
                        cons = ({'type': 'eq',
                                 'fun': lambda x: x[0]})
                    else:
                        if er1[tx] < -tol1:
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
                    for typ in [1, 2]:
                        x0 = nopt[(typ - 1) * 2:(typ - 1) * 2 + 2, tx]
                        res = minimize(util, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'disp': False})
                        result[typ - 1, tx] = res.fun
                        nopt1[(typ - 1) * 2:(typ - 1) * 2 + 2, tx] = res.x
#                        print(i, tx, typ, tst , tcnt1[i], tcnt2[i],':  ', round(res.x[0], 3), round(res.x[1], 3), 'nratio:', round(nratio, 3))
                    nuu = nopt1[0, tx]
                    nsu = nopt1[1, tx]
                    nus = nopt1[2, tx]
                    nss = nopt1[3, tx]
                for t in range(1, T):
                    nopt2[:, t] = nopt[:, t]
                    nopt[:, t] = stepsize * nopt1[:, t] + (1 - stepsize) * nopt2[:, t]
                    state(nopt[0, t], nopt[1, t], nopt[2, t], nopt[3, t], t)    
                    er2[t] = sum(abs(nopt2[:, t] - nopt1[:, t]))

# =========================== Optimization algorithm for finding the optimal set of n's =========================== #

# ===================================================== Output ===================================================== #    
                Eval = (np.log(h[160]+1) - np.log(h2000))**2 + (np.log(Pop[160]) - np.log(PopData[160]))**2 + (np.log(Y[160]) - np.log(GDPData[160]))**2
                x = range(1840, 1840 + Tmax)
                Tm = 174
                plt.plot(x[0:Tm], Pop[0:Tm], 'b', label ="Model")
                plt.plot(x[0:Tm], PopData[0:Tm], 'r', label ="Data")
                plt.xlabel('Time')
                plt.ylabel('Population (millions)')
                plt.title('Population Growth')
                axes = plt.gca()
                axes.set_xlim([1840,2010])
                plt.legend(loc=1, prop={'size':8})
                plt.xticks(np.arange(min(x), 2020, 20))
                plt.show()
                
                plt.plot(x[0:Tm], Y[0:Tm], 'b', label ="Model")
                plt.plot(x[0:Tm], GDPData[0:Tm], 'r', label ="Data")
                plt.xlabel('Time')
                plt.ylabel('GDP')
                plt.title('Output Growth')
                axes = plt.gca()
                axes.set_xlim([1840,2010])
                plt.legend(loc=1, prop={'size':8})
                plt.xticks(np.arange(min(x), 2020, 20))    
                plt.show()
                
                plt.plot(x[0:Tmax], wr[0:Tmax], 'r')
                plt.plot(x[0:Tmax], [ts/tu]*Tmax, 'g')
                plt.plot(x[0:Tmax], [ts/tu + tol1]*Tmax , 'g--')
                plt.plot(x[0:Tmax], [ts/tu - tol1 ]*Tmax, 'g--')    
                plt.xlabel('Time')
                plt.ylabel('Ratio')
                plt.title('Wages ratio')
                axes = plt.gca()
                axes.set_xlim([1840, 1840 + Tmax])
#                axes.set_ylim([1.5,2.5])
                plt.show()
                
                plt.plot(x[0:Tmax], er2[0:Tmax], 'r')
                plt.plot(x[0:Tmax], [0]*Tmax, 'g')
                plt.plot(x[0:Tmax], [tol2]*Tmax , 'g--')    
                plt.xlabel('Time')
                plt.ylabel('Error')
                plt.title('Error as the cumulative difference between two runs')
                axes = plt.gca()
                axes.set_xlim([1840, 1840 + Tmax])
                plt.xticks(np.arange(min(x), max(x), 50))
                plt.show()
                
                plt.plot(x[0:Tmax], nopt2[0, 0:Tmax], 'm--', label = "unskilled children: previous run")
                plt.plot(x[0:Tmax], nopt1[0, 0:Tmax], 'm', label = "unskilled children: current run")
                plt.plot(x[0:Tmax], nopt2[1, 0:Tmax], 'c--', label = "skilled children: previous run")
                plt.plot(x[0:Tmax], nopt1[1, 0:Tmax], 'c', label = "skilled children: current run")  
                plt.xlabel('Time')
                plt.ylabel('Number of children')
                plt.title('Number of children of the unskilled parents')
                axes = plt.gca()
                axes.set_xlim([1840, 1840 + Tmax])
                axes.set_ylim([0,2.5])    
                plt.legend(loc=1, prop={'size':8})
                plt.xticks(np.arange(min(x), max(x), 50))
                plt.show()
                
                plt.plot(x[0:Tmax], nopt2[2, 0:Tmax], 'm--', label = "unskilled children: previous run")
                plt.plot(x[0:Tmax], nopt1[2, 0:Tmax], 'm', label = "unskilled children: current run")
                plt.plot(x[0:Tmax], nopt2[3, 0:Tmax], 'c--', label = "skilled children: previous run")
                plt.plot(x[0:Tmax], nopt1[3, 0:Tmax], 'c', label = "skilled children: current run")  
                plt.xlabel('Time')
                plt.ylabel('Number of children')
                plt.title('Number of children of the skilled parents')
                axes = plt.gca()
                axes.set_xlim([1840, 1840 + Tmax])
                axes.set_ylim([0,2.5])      
                plt.legend(loc=1, prop={'size':8})
                plt.xticks(np.arange(min(x), max(x), 50))
                plt.show()
             
                plt.plot(x[0:Tmax], gammau[0, 0:Tmax], 'm', label = "unskilled parents")
                plt.plot(x[0:Tmax], gammas[0, 0:Tmax], 'c', label = "skilled parents")
                plt.plot(x[0:Tmax], [1 - alpha - beta]*Tmax , 'b--')  
                plt.xlabel('Time')
                plt.ylabel('Ratio')
                plt.title('Parenting time as a portion of the total time')
                axes = plt.gca()
                axes.set_xlim([1840, 1840 + Tmax])
                axes.set_ylim([(1-alpha-beta) - 0.2,(1-alpha-beta) + 0.2])
                plt.legend(loc=1, prop={'size':8})
                plt.xticks(np.arange(min(x), max(x), 50))
                plt.show()
               
                print(i, round(tcnt1[i]*100/T, 2),'%', round(tcnt2[i]/1.11, 2),'%')

            output('Results.xlsx', str(r1) + '-' + str(r2) + '-' + str(r3), nopt, Pop, Y, h, Eval, alpha, beta, tu, ts)
    
# ===================================================== Output ===================================================== # 