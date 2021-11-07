# -*- coding: utf-8 -*-
"""
Created on Fri Nov 09 17:02:44 2018

@author: Dan
"""

import csv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
import mi_functions_V1 as mi
import scipy.stats as stats

# This function relates H2O and SIO2, and it is determined elsewhere.
def func(x_in):
    return -0.0027811*x_in**2 + 0.24540367*x_in - 1.61493354

# New tableau colors
tableau = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ac']
# 0 - blue, 1 - orange, 2 - red, 3 - light blue, 4 - green, 5 - yellow, 6 - purple, 7 - pink, 8 - brown, 9 - gray


Peq = []
Pentrap = []
d_mi = []
mi_samples = []

flag = 1
with open('input\\test_mi_data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if flag == 1:
            mi_headers = [ s.upper() for s in row[1:] ]+['H2Ocalc']
            flag = 0
        else:
            mi_samples.append(row[0])
            d_mi.append([ float(x) if x != '' else np.nan for x in row[1:] ])
            H2O = d_mi[-1][mi_headers.index('H2O')]
            if H2O > 0:
                temp,err = mi.VolatileCalc('sp','basalt',[H2O,0,49,1100])
            if temp[0] > 0:
                Peq.append(temp[0])
            H2O = func( d_mi[-1][mi_headers.index('SIO2')] )
            
            # STOP NEGATIVE VALUES
            if H2O < 0.2:
                H2O = 0.2
                
            CO2 = d_mi[-1][mi_headers.index('CO2')]
            if H2O > 0:
                temp,err = mi.VolatileCalc('sp','basalt',[H2O,CO2,49,1100])
                if temp[0] > 0:
                    temp_pentrap = temp[0]
                else:
                    temp_pentrap = 0
            else:
                temp_pentrap = 0
            if temp_pentrap > 0:
                Pentrap.append(temp_pentrap)
            H2O = func( d_mi[-1][mi_headers.index('SIO2')] )
            
            # STOP NEGATIVE VALUES
            if H2O < 0.2:
                H2O = 0.2
                
            d_mi[-1] += [H2O,temp_pentrap]

d_miT = list(map(list, zip(*d_mi)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% Degassing path
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The parameters of this calculation must be determined elsewhere.
dp, err = mi.VolatileCalc('dp','basalt',[max(d_miT[-2])+0.1,2000,49,1100,[0],500]) 

dp_T = list(map(list, zip(*dp)))
H2O = min(dp_T[1])
step = 0.1

while H2O - step > 0:
    out,err = mi.VolatileCalc('sp','basalt',[H2O-step,0,49,1100])
    dp.append(out)
    H2O -= step
dp.append([0]*7)

dp_T = list(map(list, zip(*dp)))

x = dp_T[0]
y = dp_T[2]
calc_CO2 = sp.interpolate.interp1d(x,y, kind='linear')
x = dp_T[0]
y = dp_T[1]
calc_H2O = sp.interpolate.interp1d(x,y, kind='linear')

#%%
const = 0 #1 to take log of K2O in SO2-K2O regression
conf = 0.68
MC = 50

H2Oi = 0
CO2i = 0
K2Oi = 0
SO2i = 0
K2Oii = 0
outputs = []
Kdlist = []
Plist = []
CO2vap = []

prof = mi.define_profile(12,2000)

for i in range(MC):
    print (i)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Resample melt inclusion composition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    resample_d = [[],[],[],[]] #SiO2, K2O, S, H2O
    count = 0
    for ii in ['SIO2','K2O','S']:
        if MC > 1:
            for center,err in zip(d_miT[mi_headers.index(ii)],d_miT[mi_headers.index(ii+'ERR')]):
                if center > 0:
                    if not err > 0:
                        err = center * 0.05
                    if ii == 'S':
                        temp = np.random.normal(loc=center * 64.066/32.065, scale=err * 64.066/32.065, size=None)
                    else:
                        temp = np.random.normal(loc=center, scale=err, size=None)
                    resample_d[count].append(temp)
                else:
                    resample_d[count].append(np.nan)
        else:
            if ii == 'S':
                resample_d[count] = np.array(d_miT[mi_headers.index(ii)]) * 64.066/32.065
            else:
                resample_d[count] = d_miT[mi_headers.index(ii)]
        count += 1
                
    for ii in resample_d[0]:
        resample_d[3].append( func(ii) )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # H2O vs. K2O
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if i == 0:
        grid_size = (1,3)
        ax =  plt.subplot2grid(grid_size, (0,0), rowspan=1, colspan=1)
        x = np.array(d_miT[mi_headers.index('SIO2')])
        y = func(x)
        x = np.array(d_miT[mi_headers.index('K2O')])
        ax.set_xlabel('$\mathregular{K_2O}$ (wt%)',fontweight='bold')
        ax.set_ylabel('$\mathregular{H_2O}$ (wt%)',fontweight='bold')
        ax.set_xlim(0,6)
        ax.set_ylim(0,4)
        ax.scatter(x,y,marker = 'o',color = 'k',s=60,zorder = 100)

        ax.vlines(x,func(np.array(d_miT[mi_headers.index('SIO2')])-np.array(d_miT[mi_headers.index('SIO2ERR')])),func(np.array(d_miT[mi_headers.index('SIO2')])+np.array(d_miT[mi_headers.index('SIO2ERR')])),zorder= 150, color = 'k',lw = 2)
        x_err = np.array(d_miT[mi_headers.index('K2OERR')])
        ax.hlines(y,x-x_err,x+x_err,zorder= 150, color = 'k',lw = 2)
        
        
    #H2O vs. K2O
    x = resample_d[1]
    y = resample_d[3]
    
    ax.scatter(x,y,marker='o',c='darkgrey',alpha=1,zorder=20,s=8,lw=0)
    
    # Regression
    p, cov = np.polyfit(x, y, 1, cov=True)
    y_model = np.polyval(p, x)
    
    p1_K2O_H2O = p[0]
    p2_K2O_H2O = p[1]
    
    # Statistics
    n = len(y)
    m = p.size
    DF = n - m
    t = stats.t.ppf( (1+conf)/2 , n - m )
    
    resid = y - y_model                           
    chi2 = np.sum((resid/y_model)**2)
    chi2_red = chi2/(DF)
    s_err = np.sqrt(np.sum(resid**2)/(DF))
    
    x2 = np.arange(0.1,6.1,0.1)
    y2 = np.polyval(p, x2)
    
    
    ax.plot(x2,y2,"-", color=tableau[2], linewidth=1, alpha=0.5, label='Fit',zorder=50)  
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # K2O vs. SO2
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if i == 0:
        ax2 =  plt.subplot2grid(grid_size, (0,1), rowspan=1, colspan=1)
        x = np.array(d_miT[mi_headers.index('K2O')])
        y = np.array(d_miT[mi_headers.index('S')]) * 64.066/32.065
        ax2.scatter(x,y*32/64.,marker = 'o',color = 'k',s=60,zorder = 100)
        
        y_err = np.array(d_miT[mi_headers.index('SERR')])
        ax2.vlines(x,y*32/64.-y_err, y*32/64.+y_err,zorder= 150, color = 'k',lw = 2)
        x_err = np.array(d_miT[mi_headers.index('K2OERR')])
        ax2.hlines(y*32/64.,x-x_err,x+x_err,zorder= 150, color = 'k',lw = 2)

        ax2.set_xlabel('$\mathregular{K_2O}$ (wt%)',fontweight='bold')
        ax2.set_ylabel('$\mathregular{S}$ (ppm)',fontweight='bold')
        ax2.set_xlim(0,3)
        ax2.set_ylim(0,3000)
    x = np.array(resample_d[1])
    y = np.array(resample_d[2])
    
    x = x[y>0]
    y = np.log(y[y>0])
    
    ax2.scatter(x,np.exp(y)*32/64.,marker='o',c='darkgrey',alpha=1,zorder=20,s=8,lw=0)
    
    # Regression
    p, cov = np.polyfit(x, y, 1, cov=True)
    y_model = np.polyval(p, x)
    
    p1_K2O_SO2 = p[0]
    p2_K2O_SO2 = p[1]
    
    # Statistics
    n = len(y)
    m = p.size
    DF = n - m
    t = stats.t.ppf( (1+conf)/2 , n - m )
    
    resid = y - y_model                           
    chi2 = np.sum((resid/y_model)**2)
    chi2_red = chi2/(DF)
    s_err = np.sqrt(np.sum(resid**2)/(DF))
    
    x2 = np.arange(0.01,6.01,0.1)
    y2 = np.polyval(p, x2)
    
    ax2.plot(x2,np.exp(y2)*32/64.,"-", color=tableau[3], linewidth=1, alpha=0.5, label="Fit",zorder=50)  
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Model
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    out = [[] for z in range(18)]
    CO2out = []
    
    flag = 1
    flag2 = 1
    Kdlist.append([])
    #Start at 12 km depth
    for ii in np.linspace(12,0.01,240): #max(resample_d[3])
        i_depth = min(range(len(prof[0])), key=lambda j: abs(prof[0][j]-ii))
        P = prof[3][i_depth]
        
        H2O = calc_H2O(P)
        CO2 = calc_CO2(P)
        K2O = (H2O - p2_K2O_H2O) / p1_K2O_H2O
        
        #Density
        if K2O <= 2.2:
            p = -1.987E-3*K2O**3 + 3.990E-02*K2O**2 + -1.628E-1*K2O + 2.504
        else:
            p = -1.987E-3*2.2**3 + 3.990E-02*2.2**2 + -1.628E-1*2.2 + 2.504
        
        #How does S change with depth?
        if const == 1:
            SO2 = np.exp( p1_K2O_SO2 * np.log(K2O) + p2_K2O_SO2 )
        else:
            if H2O <= -2.:
                SO2 = SO2i
            else:
                #SO2 is a function of melt K2o
                SO2 = np.exp( p1_K2O_SO2 * K2O + p2_K2O_SO2 )
        if flag == 0:
            F = K2Oi/K2O
            Ftot = K2Oii/K2O
            Xvh2o = F*((H2Oi*(K2O/K2Oi)-H2O)/100.)
            Xvso2 = F*((SO2i*(K2O/K2Oi)-SO2)/1000000.)
            Xvco2 = F*((CO2i*(K2O/K2Oi)-CO2)/1000000.)
            Xvapor = Xvh2o+Xvso2+Xvco2
            Xcrystal = 1-F-Xvapor
            Mvapor = Xvapor / (1-F)
            D = 1 + np.log(SO2/SO2i)/np.log(F)
            Kd = D/Mvapor
            Kdlist[-1].append(Kd)
            
            #Calculate vapor
            H2Om = Xvh2o / 18.01528
            SO2m = Xvso2 / 64.066
            CO2m = Xvco2 / 44.01
            
            H2Omf = H2Om / (H2Om + SO2m + CO2m)
            SO2mf = SO2m / (H2Om + SO2m + CO2m)
            CO2mf = CO2m / (H2Om + SO2m + CO2m)
            
            HS = H2Omf/SO2mf
            CS = CO2mf/SO2mf
            
            #Mass fraction of SO2 in the vapor
            Svap = Xvso2/(Xvh2o+Xvso2+Xvco2)
            
            #Mass fraction of H2O in the vapor
            Hvap = Xvh2o/(Xvh2o+Xvso2+Xvco2)
            
            #Mass fraction of CO2 in the vapor
            Cvap = Xvco2/(Xvh2o+Xvso2+Xvco2)
            
            #Mass fraction of SO2 in the vapor
            CO2out.append(Xvco2/(Xvh2o+Xvso2+Xvco2))
            
            #Output results
            if i == 0:
                Plist.append(P)
            out[0].append(P)
            out[1].append(SO2) #*32.065/64.066
            out[2].append(H2O)
            out[3].append(CO2)
            out[4].append(SO2mf)
            out[5].append(H2Omf)
            out[6].append(CO2mf)
            out[7].append((H2O/18.01528)/(SO2/(10000*64.066))) #This is H2O/SO2
            out[8].append(CS)
            out[9].append(Mvapor)
            out[10].append(Kd)
            out[11].append(K2O)
            out[12].append(Svap)
            out[13].append(F)
            out[14].append(Ftot)
            out[15].append(p)
            out[16].append(Hvap)
            out[17].append(Cvap)
        else:
            K2Oii = K2O
        H2Oi = H2O
        CO2i = CO2
        K2Oi = K2O
        SO2i = SO2
        flag = 0
    if i == 0:
        ax3 =  plt.subplot2grid(grid_size, (0,2), rowspan=1, colspan=1)
        x = np.array(d_miT[mi_headers.index('SIO2')])
        x = func(x)
        y = np.array(d_miT[mi_headers.index('S')]) * 64.066/32.065
        
        y_err = np.array(d_miT[mi_headers.index('SERR')])
        ax3.vlines(x,y*32/64.-y_err, y*32/64.+y_err,zorder= 150, color = 'k',lw = 1)
        ax3.hlines(y*32/64.,func(np.array(d_miT[mi_headers.index('SIO2')])-np.array(d_miT[mi_headers.index('SIO2ERR')])),func(np.array(d_miT[mi_headers.index('SIO2')])+np.array(d_miT[mi_headers.index('SIO2ERR')])),zorder= 150, color = 'k',lw = 1)

        ax3.scatter(x,y*32/64.,marker = 'o',color = 'k',s=60,zorder = 100)
        #ax3.scatter(3,200,color='yellow',edgecolors='k',lw=1,marker='*',s=100,zorder=500)
        ax3.set_xlabel('$\mathregular{H_2O}$ (wt%)',fontweight='bold')
        ax3.set_ylabel('$\mathregular{S}$ (ppm)',fontweight='bold')
        ax3.set_xlim(0,5)
        ax3.set_ylim(0,3000)
    ax3.scatter(resample_d[3],np.array(resample_d[2])*32/64.,marker='o',c='darkgrey',alpha=1,zorder=20,s=8,lw=0)
    y_temp = np.array(out[1])
    ax3.plot(out[2],y_temp*32/64., color=tableau[2], linewidth=1, alpha=0.5, label="Model",zorder=50)
    outputs.append(out)
    CO2vap.append(CO2out)

plt.tight_layout(rect = [0,0,1,1], pad = 0, w_pad = -1.5, h_pad = 0)
plt.gcf().set_facecolor('white')

fig = plt.gcf()
fig.set_size_inches(12,3.2, forward = True)
plt.savefig("output\\test_DP_MC50_Regressions.png", dpi = 300, bbox_inches ='tight') #, dpi = 300
plt.show()


#%% Outputs

out_data = [['P', 'S', 'H2O', 'CO2', 'SO2mf-vap', 'H2Omf-vap', 'CO2mf-vap','H2O/SO2-melt','H2O/SO2-vap', 'Mvapor', 'Kd', 'K2O','SO2-MassFracVap','F','Ftot','p','']]
for i in out_data[0][:-1]:
    out_data[0].append(i+'err')
outputs_T = list(map(list, zip(*outputs)))

count = 0
temp1 = []
temp2 = []
for i in outputs_T:
    temp1.append([])
    temp2.append([])
    i_T = list(map(list, zip(*i)))
    for ii in i_T:
        if count == 1:
            ii = np.array(ii)
            ii *= 32.065/64.066
        loc_param, scale_param = stats.norm.fit(ii)
        temp1[-1].append(loc_param)
        temp2[-1].append(scale_param)
    count+=1
temp1_T = list(map(list, zip(*temp1)))
temp2_T = list(map(list, zip(*temp2)))

for i in range(len(temp1_T)):
    out_data.append(temp1_T[i]+['']+temp2_T[i])

with open('output\\test_DP_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(out_data)

#%% PLOT MODEL

font_size = 20
prof = mi.define_profile(20,5000)
def func(x_in):
    return -0.0027811*x_in**2 + 0.24540367*x_in - 1.61493354

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PLOT A
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
colors = (tableau[0], tableau[2], tableau[4],tableau[0], tableau[2], tableau[4],'k')

grid_size = (1,3)
ax = plt.subplot2grid(grid_size, (0,0), rowspan=1, colspan=1)

axes = [ax, ax.twiny(), ax.twiny()]
axes[-1].spines['top'].set_position(('axes', 1.18))
axes[-1].set_frame_on(True)
axes[-1].patch.set_visible(False)

axes[0].set_ylabel('P (MPa)', fontsize = font_size*1.2, fontweight = 'bold')
axes[0].set_ylim([259.697,0])
axes[0].set_xlim([0,4.0])
axes[0].set_xticks(range(0,5,1))
axes[1].set_xlim([0,1200])
axes[1].set_xticks(range(0,1600,400))
axes[2].set_xlim([0,2400])
axes[2].set_xticks(range(0,3000,600))

axes[0].tick_params(axis='both', which='major', labelsize=font_size, length=10, width = 1.5, pad = 5,zorder=0)
axes[1].tick_params(axis='both', which='major', labelsize=font_size, length=10, width = 1.5, pad = 5,zorder=0)
axes[2].tick_params(axis='both', which='major', labelsize=font_size, length=10, width = 1.5, pad = 5,zorder=0)

labels = ('$\mathregular{H_{2}O}$ (wt%)', '$\mathregular{CO_{2}}$ (ppm)', 'S (ppm)')

count = 1
order = [2,3,1]
order_mi = [-2, mi_headers.index('CO2'),mi_headers.index('S'),mi_headers.index('CO2ERR'),mi_headers.index('SERR')]
for ax, color in zip(axes, colors):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_zorder(-200)
    ax.tick_params(axis='x', colors=color)
for ax, color in zip(axes, colors):
    if count == 3:
        end = len(outputs)
        lw = 1
        alph = 0.5
    else:
        end = 1
        lw = 2.5
        alph = 1
    #Plot degassing path
    for out in outputs[:end]:
        if count == 3:
            factor = 32.065/64.066
        else:
            factor = 1
        ax.plot(np.array(out[order[count-1]])*factor,out[0], linestyle='--', color=color, lw = lw, dashes=(4, 1.5),alpha=alph,zorder=0)
    
    if count == 1:
        ax.set_xlabel(labels[count-1], color=color, fontsize = font_size*1.2, fontweight = 'bold', labelpad = 3)
    else:
        ax.set_xlabel(labels[count-1], color=color, fontsize = font_size*1.2, fontweight = 'bold', labelpad = 5)
    ax.scatter(d_miT[order_mi[count-1]],d_miT[-1],color=color,marker='o',s=70,lw=1.5,zorder=200,edgecolors='k',clip_on=False)
    if count == 1:
        xerr1 = func(np.array(d_miT[mi_headers.index('SIO2')])-np.array(d_miT[mi_headers.index('SIO2ERR')]))
        xerr2 = func(np.array(d_miT[mi_headers.index('SIO2')])+np.array(d_miT[mi_headers.index('SIO2ERR')]))
        ax.hlines(d_miT[-1],xerr1,xerr2,lw=1.5,zorder=100,color='k')
    else:
        xerr = np.array(d_miT[order_mi[count+1]])
        ax.hlines(d_miT[-1],d_miT[order_mi[count-1]]-xerr,d_miT[order_mi[count-1]]+xerr,lw=1.5,zorder=100,color='k')
    
    count += 1
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PLOT B
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ax = plt.subplot2grid(grid_size, (0,1), rowspan=1, colspan=1)

axes = [ax, ax.twiny(), ax.twiny()]
axes[-1].spines['top'].set_position(('axes', 1.18))
axes[-1].set_frame_on(True)
axes[-1].patch.set_visible(False)
axes[-1].set_zorder(0)
axes[1].set_zorder(1)

colors = (tableau[0], tableau[2], tableau[4],tableau[0], tableau[2], tableau[4],'k')
labels = ('$\mathregular{H_{2}Ov}$ (mol%)', '$\mathregular{CO_{2}v}$ (mol%)', '$\mathregular{SO_{2}v}$ (mol%)')

count = 1
order = [5,6,4]
for ax, color in zip(axes, colors):
    if count in range(4):
        end = len(outputs)
        z = 0
        alph = 0.1
    else:
        end = 1
        z = 10
        alph = 1
    for out in outputs[:end]:
        ax.plot(np.array(out[order[count-1]])*100,out[0], linestyle='-', color=color, lw = 2.5, zorder = z,alpha = alph)
    
    ax.tick_params(axis='x', colors=color)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    if count == 1:
        ax.set_xlabel(labels[count-1], color=color, fontsize = font_size*1.2, fontweight = 'bold', labelpad = 3)
    else:
        ax.set_xlabel(labels[count-1], color=color, fontsize = font_size*1.2, fontweight = 'bold', labelpad = 5)
    count += 1
axes[0].set_ylim([259.697,0])


axes[0].set_xlim([85,100])
axes[0].set_xticks(range(85,105,5))

axes[1].set_xlim([0,8])
axes[1].set_xticks(range(0,10,2))
axes[2].set_xlim([0,8])
axes[2].set_xticks(range(0,10,2))
'''
for i in range(3):
    axes[i].set_xlim([0,100])
    axes[i].set_xticks(range(0,125,25))
'''

axes[0].tick_params(axis='both', which='major', labelsize=font_size, length=10, width = 1.5, pad = 5)
axes[1].tick_params(axis='both', which='major', labelsize=font_size, length=10, width = 1.5, pad = 5)
axes[2].tick_params(axis='both', which='major', labelsize=font_size, length=10, width = 1.5, pad = 5)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(1.5)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PLOT C
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ax = plt.subplot2grid(grid_size, (0,2), rowspan=1, colspan=1)
for out in outputs:
    ax.plot(out[7],out[0], linestyle='-', color=colors[-1], lw = 1.0, alpha = 0.3,label='Vapor')

ax.set_xlabel('$\mathregular{H_{2}O/SO_2}$', fontsize = font_size*1.2, fontweight = 'bold', labelpad = 3)
ax.set_xlim(10,10**4)

ax.set_ylim([259.697,0])
ax.set_xscale('log')

plt.tick_params(axis='x', which='major', labelsize=font_size, length=10, width = 1.5, pad = 7)
plt.tick_params(axis='y', which='major', labelsize=font_size, length=10, width = 1.5, pad = 5)
plt.tick_params(axis='both', which='minor', length=5, width = 1.0, pad = 5)


plt.gcf().set_facecolor('white') 
fig = plt.gcf()
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(1.5)


#Secondary axis
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
y2labels = []
y2ticks = []
for i in range(0,12,2):
    y2labels.append(i)
    i_depth = min(range(len(prof[0])), key=lambda j: abs(prof[0][j]-i))
    y2ticks.append(prof[3][i_depth])
ax2.set_yticks(y2ticks)
ax2.set_yticklabels(y2labels)
ax2.set_ylabel('Depth (km)', color='k', fontsize = font_size*1.2, fontweight = 'bold', labelpad = 5)
plt.tick_params(axis='y', which='major', labelsize=font_size, length=10, width = 1.5, pad = 10)


fig.set_size_inches(14,7, forward = True)
plt.savefig("output\\test_figure.pdf", dpi = 300, bbox_inches ='tight') #, dpi = 300
plt.show()

