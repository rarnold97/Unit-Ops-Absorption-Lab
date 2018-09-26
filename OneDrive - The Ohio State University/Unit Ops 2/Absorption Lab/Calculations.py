# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:25:37 2018
Calculations for Absorption Unit ops report 
@author: ryanm
"""

import pandas as pd
import numpy as np
import interpolate as ip

import matplotlib as mpl
import matplotlib.pyplot as plt

from textwrap import wrap


from pandas import ExcelWriter
from pandas import ExcelFile

import xlsxwriter as xl

#define constants

ch2o = 55.5 # M
Dab_co2air = 0.177/(10**4)  # m^2/s 317.2K Middleman
Dab_co2water = 2.0*10**(-9) # %m^2/s 307.2 Middleman
Acol = (np.pi/4)*(0.15)**2  # Column Area m^2
z = 1.3 #packing height in m
rho_air = 1.184 #kg/m^3
rho_co2 = 1.784 #kg/m^3
rho_water = 1 #kg/m^3
Y_co2_ambient = 500 / 10**6 #assuming ambient CO2 composition is 500 ppm
R = 0.08206 #Ideal gas constant l*atm/mol*K

#Raschig Ring Information
ID = 11  #mm
OD = 16  #mm 
h = 12.7  #mm 
OSA = 2*np.pi*(OD/2)*(h)  #m^2
ISA = 2*np.pi*(ID/2)*(h)  #m^2
area_top = (np.pi*(OD/2)**2) - (np.pi*(ID/2)**2) # %m^2
SA = OSA + ISA + 2*area_top # mm^2 
SA_m = SA / (1000**2)  # m^2 
e = 0.64 # Void space coefficient for ceramic rashig ring of 1/2 in. size
#end of Rashig Ring information

#viscosity information
u_air = (ip.interp(10,37,1.78,1.90,25))/10**5 #From Geankopolis kg/m*s
u_co2 = ip.interp(283.2,311.0,0.0141,0.0154,298) / 10**3 #From Geankopolis kg/m*s
u_water = 8.9 * 10**(-4) #kg/m*s 1 atm 25 C assume incompressibility retrieved from engineering toolbox
u_water_vapor = 9.9*10**(-6) # kg/m*s 1 atm 25 C assume retrieved from engineering toolbox assuming does not change much within a degree celsius 


data = pd.read_excel('Absorption Lab.xlsx', sheetname='CO2 Absorption') #Read in excel spreadsheet

trial_list = [0,2,5,7,9,12] #define the indecies that are for the relevant trials 
index = [0, 1, 2, 3, 4, 5 ] #initialize the index for the desired spliced column data 

X_co2out_all = np.array(data['XCO2out']) #read in the column for output gas co2 concentrations 

X_co2out = np.zeros(6) #allocate memory to the array
for i,j in zip(trial_list, index):
    X_co2out[j] = X_co2out_all[i]  #splice out desired values 


Y_co2in_all = np.array(data['Gas in Concentration %'])

Y_co2in = np.zeros(6)

for i,j in zip(trial_list, index):
    Y_co2in[j] = (Y_co2in_all[i]/100) #values in the spreadsheet are in units of percent must divide by 100 
    
Ptot_all = np.array(data['Relative Pressure (in water)'])

Ptot = np.zeros(6)

for i, j in zip(trial_list, index):
    Ptot[j] = (Ptot_all[i] * 0.00245586) + 1 #read in relative pressures and convert them to pressure absolute

vp_h2o = np.array([0.03247407,0.032283278,0.03247407,0.03247407,0.03247407,0.032686061]) # values of vapor pressure Geankopolis
Y_h2oout = vp_h2o / Ptot

Y_co2out_all = np.array(data['Gas out concentration %'])

Y_co2out = np.zeros(6)

for i,j in zip(trial_list, index):
    Y_co2out[j] = (Y_co2out_all[i] / 100) #convert the percentage to fraction while splcing the relevant trial data

Qdot_water_all = np.array(data['Water flow Rate (L/min)']) # L/min

Qdot_water = np.zeros(6)

for i, j in zip(trial_list, index):
    Qdot_water[j] = Qdot_water_all[i] # L/min
    
Qdot_air_all = np.array(data['Air Flow Rate (L/min)']) #L/min

Qdot_air = np.zeros(6)

for i, j in zip(trial_list, index):
    Qdot_air[j] = Qdot_air_all[i] #L/min


    
T_xin_all = np.array(data['Liquid in temp'])

T_xin = np.zeros(6)

for i,j in zip(trial_list, index):
    T_xin[j] = T_xin_all[i]
    
T_xout_all = np.array(data['Liquid out temp'])

T_xout = np.zeros(6)

for i,j in zip(trial_list, index):
    T_xout[j] = T_xout_all[j]
    
T_yin_all = np.array(data['Gas in temp celsius'])

T_yin = np.zeros(6)

for i,j in zip(trial_list, index):
    T_yin[j] = T_yin_all[i] 

T_yout_all = np.array(data['Gas out temp'])
T_yout = np.zeros(6)

for i,j in zip(trial_list, index):
    T_yout[j] = T_yout_all[i]
    

#Mixture Densities    
Vmixin = (1 - Y_co2in)*0.029/rho_air + Y_co2in*0.044/rho_co2    
Vmixout = (1 - Y_co2out - Y_h2oout)*0.029/rho_air + Y_co2out*0.044/rho_co2 + Y_h2oout*0.018/rho_water
rho_mix_yin = ((1 - Y_co2in)*0.029 +  (Y_co2in*0.044))/Vmixin  #kg/m^3
rho_mix_yout = ((1 - Y_co2out - Y_h2oout)*0.029 + Y_co2out*0.044 + Y_h2oout*0.018)/Vmixout #kg/m^3

#Pure Component Mass Flow rates
L_prime = ((rho_water/1000)*(Qdot_water))/60 #kg/s
V_prime = ((rho_mix_yin/1000)*(Qdot_air))/60 #kg/s

L_prime_mol = L_prime/0.018 #Mol/s
V_prime_mol = V_prime/0.029

L = L_prime / (1-X_co2out)
#Mass Balance for X_co2in

#alpha = (X_co2out/(1-X_co2out)) + (V_prime / L_prime)*((Y_co2out/(1-Y_co2out))-(Y_co2in/(1-Y_co2in)))

#X_co2in_mb = alpha/(1+alpha)

#Calculation of Xin using Henrys law assuming 500 ppm ambient conditions 

H_co2in = ip.interp(293.2,303.2,1420,1860,(T_xin+273.2)) 
H_co2out = ip.interp(293.2,303.2,1420,1860,(T_xout+273.2))

H = (H_co2in + H_co2out)/2
X_co2_hl = (Y_co2_ambient)/H_co2out 
X_co2in = 0 #Assume this because values are negative 
#Calculation of mix viscosities

u_mix_yin = (Y_co2in*u_co2*np.sqrt(0.044) + (1-Y_co2in)*u_air*np.sqrt(0.029)) / (Y_co2in*np.sqrt(0.044) + (1-Y_co2in)*np.sqrt(0.029))
u_mix_yout = (Y_co2out*u_co2*np.sqrt(0.044) + (1-Y_co2out-Y_h2oout)*u_air*np.sqrt(0.029) + Y_h2oout*u_water_vapor*np.sqrt(0.018)) / (Y_co2out*np.sqrt(0.044) + \
              (1-Y_co2out-Y_h2oout)*np.sqrt(0.029) + Y_h2oout*np.sqrt(0.018)  )

u_air_276 = 17.29*10**-6 #Kg/m*s engineering tool box air viscosity at 1 atm 276K
u_air_298 = 18.36*10**-6 #Kg/m*s engineering tool box air viscosity at 1 atm 298K
u_water_307 = 0.0007335 #Kg/m*s Engineering tool box water viscosity at 1 atm 307.2

# non pure component mass flow rates for the vapor streams 
V2 = (V_prime_mol) / (1-Y_co2in)
V1 = (V_prime_mol) / (1 - Y_co2out - Y_h2oout)
V_av = (V1+V2)/2

#Average  of some physical properties.  Not necessary for liquid since compositions are so dilute
rho_yav = (rho_mix_yin+ rho_mix_yout)/2
u_mix_V_av = (u_mix_yin+ u_mix_yout)/2
X_h2o = 1 #not totally accurate but a very valid assumption

#interstitial velocities 
Ux_L = (L/rho_water) / Acol
Ux_V = 0.029*(V_av / rho_yav) / Acol #pay attention to units here

#Superficial Velocities
U_prime_L = e * Ux_L 
U_prime_V = e * Ux_V

#Reynolds Numbers 
dp = 0.567 * np.sqrt(SA_m)
a = (6*(1-e)) / dp

Re_L = (rho_water * U_prime_L*dp) / u_water
Re_V = (rho_yav * U_prime_V * dp) / u_mix_V_av

#Chilton and Colburn Factors
jD_L = ((0.765/Re_L**0.82)+(0.365/Re_L**0.386))*(1/e)
jD_V = ((0.765/Re_V**0.82)+(0.365/Re_V**0.386))*(1/e)

#Schmidt Numbers 
Dab_airco2 = ((1.42*10**-5)*298*u_air_276 )/(276*u_air_298)
Dab_waterco2 = ((0.198/10**4)*298*u_water_307 ) / (307.2*u_water)

Sc_L = u_water / (rho_water *Dab_co2water ) #not in the valid range for whatever reason

Sc_V = u_mix_V_av / (rho_yav * Dab_airco2)

#Stanton Number Liquid
St = jD_L / (Sc_L**(2/3))

#F factor liquid
F = St * ch2o * Ux_L

# k*c Vapor
kc_star = (jD_V * U_prime_V) / (Sc_V**(2/3)) 

#Mass Transfer Coefficients Vapor
T_yav = (T_yin + T_yout) / 2 
ky = kc_star * (Ptot/(R*T_yav))

#kx from liquid
kx = F / X_h2o

#note due to linear equilibrium relationship, we will assume that 
Ky = 1 / (1/ky + 1/(H*kx)) 

#Vapor
HOG = V_av / (Ky*a*Acol) #Assuming [(1/y)*M]av is 1 due to linear nature of equilibrium line
NOG = z / HOG
#Liquid
Kx = 1/( 1/(H*ky) + 1/kx) #Convert H to a molar basis

HOL = L_prime_mol/ (Kx * a * Acol)
NOL = z/HOL

#Calculate the Experimental values
#Equilibrium Values
Y_iout = (H * X_co2in)
X_iout = (Y_co2in*Ptot) / H
Y_iin = (H* X_co2out)
X_iin = (Y_co2out*Ptot) /H
#log means
Y_M = ( (Y_co2in - Y_iin) - (Y_co2out - Y_iout)) / np.log((Y_co2in - Y_iin)/(Y_co2out - Y_iout))

X_M =  (X_iout - X_co2out) - (X_iin - X_co2in) / np.log((X_iout - X_co2out)/(X_iin - X_co2in)) 

#Transfer Units
NOG_exp = (Y_co2in - Y_co2out) / Y_M

NOL_exp = (X_co2out - X_co2in)/X_M

HOG_exp = z / NOG_exp

HOL_exp = z / NOL_exp

Ky_exp = V_av / (HOG_exp * a * Acol)

Kx_exp = L_prime / (HOL_exp *a * Acol)

#Efficiency of Column
eta = 1 - (Y_co2out/Y_co2in)*(V1/V2)

#Average the trials

Qdot_water_av = np.zeros(3)
HOG_av = np.zeros(3)
HOL_av = np.zeros(3)
NOG_av = np.zeros(3)
NOL_av = np.zeros(3)
Ky_av = np.zeros(3)
Kx_av = np.zeros(3)

HOG_exp_av = np.zeros(3)
HOL_exp_av = np.zeros(3)
NOG_exp_av = np.zeros(3)
NOL_exp_av = np.zeros(3)
Ky_exp_av = np.zeros(3)
Kx_exp_av = np.zeros(3)
eta_av = np.zeros(3)

HOG_exp_err = np.zeros(3)
HOL_exp_err = np.zeros(3)
NOG_exp_err = np.zeros(3)
NOL_exp_err = np.zeros(3)
Ky_exp_err = np.zeros(3)
Kx_exp_err = np.zeros(3)

HOG_err = np.zeros(3)
HOL_err = np.zeros(3)
NOG_err = np.zeros(3)
NOL_err = np.zeros(3)
Ky_err = np.zeros(3)
Kx_err = np.zeros(3)

for i in range(0,3,1):
    Qdot_water_av[i] = (Qdot_water[i] + Qdot_water[i+3])/2
    HOG_av[i] = (HOG[i] + HOG[i+3])/2
    HOG_err[i] = np.std([HOG[i],HOG[i+3]]) * 12.71/np.sqrt(2)
    
    HOL_av[i] = (HOL[i]+HOL[i+3])/2
    HOL_err[i] = np.std([HOL[i],HOL[i+3]]) * 12.71/np.sqrt(2)
    
    NOL_av[i] = (NOL[i]+NOL[i+3]) /2
    NOL_err[i] = np.std([NOL[i],NOL[i+3]]) * 12.71/np.sqrt(2)
    
    NOG_av[i] = (NOG[i] +NOG[i+3]) /2
    NOG_err[i] = np.std([NOG[i],NOG[i+3]]) * 12.71/np.sqrt(2)
    
    Ky_av[i] = (Ky[i] + Ky[i+3])/2
    Ky_err[i] = np.std([Ky[i],Ky[i+3]]) * 12.71/np.sqrt(2)
    
    Kx_av[i] = (Kx[i] + Kx[i+3]) /2
    Kx_err[i] = np.std([Kx[i],Kx[i+3]]) * 12.71/np.sqrt(2)
    
    eta_av[i] = (eta[i] + eta[i+3]) / 2 
    
    HOG_exp_av[i] = (HOG_exp[i] + HOG_exp[i+3])/2
    HOG_exp_err[i] = np.std([HOG_exp[i] , HOG_exp[i+3]]) * 12.71/np.sqrt(2)
    
    HOL_exp_av[i] = (HOL_exp[i]+HOL_exp[i+3])/2
    HOL_exp_err[i] = np.std([HOL_exp[i] , HOL_exp[i+3]]) * 12.71/np.sqrt(2)
    
    NOL_exp_av[i] = (NOL_exp[i]+NOL_exp[i+3]) /2
    NOL_exp_err[i] = np.std([NOL_exp[i] , NOL_exp[i+3]]) * 12.71/np.sqrt(2)
    
    NOG_exp_av[i] = (NOG_exp[i] +NOG_exp[i+3]) /2
    NOG_exp_err[i] = np.std([NOG_exp[i] , NOG_exp[i+3]]) * 12.71/np.sqrt(2)
    
    Ky_exp_av[i] = (Ky_exp[i] + Ky_exp[i+3])/2
    Ky_exp_err[i] = np.std([Ky_exp[i] , Ky_exp[i+3]]) * 12.71/np.sqrt(2)
    
    Kx_exp_av[i] = (Kx_exp[i] + Kx_exp[i+3]) /2
    Kx_exp_err[i] = np.std([Kx_exp[i] , Kx_exp[i+3]]) * 12.71/np.sqrt(2)
    
    
    
fig1 = plt.figure(figsize=(8,6))

plt.grid(color='k',linestyle='-',linewidth=1,alpha=0.1)
#plt.axis([5,8,0,12])
plt.plot(Qdot_water_av,HOG_av,'r',marker='o',label="Empirical")
plt.errorbar(Qdot_water_av,HOG_av, yerr=HOG_err, Linestyle = "None")
plt.plot(Qdot_water_av,HOG_exp_av,'--k',marker='x',label="Experimental")
plt.errorbar(Qdot_water_av,HOG_exp_av, yerr=HOG_exp_err, Linestyle = "None")
plt.title('Height of Gas Phase Mass Transfer Units vs. Liquid Flow Rate \n')
plt.xlabel('Liquid Flow Rate L/min.')
plt.ylabel('(m)')
plt.legend(loc='best')
plt.show()

fig1.savefig('HOG.pdf')

fig2 = plt.figure(figsize=(8,6))

plt.grid(color='k',linestyle='-',linewidth=1,alpha=0.1)
#plt.axis([5,8,0,5])
plt.plot(Qdot_water_av, HOL_av, 'r',marker='o',label="Empirical")
plt.errorbar(Qdot_water_av,HOL_av, yerr=HOL_err, Linestyle = "None")
plt.plot(Qdot_water_av,HOL_exp_av,'--k',marker = 'x',label="Experimental")
plt.errorbar(Qdot_water_av,HOL_exp_av, yerr=HOL_exp_err, Linestyle = "None")
plt.title('Height of Liquid Phase Mass Transfer Units vs. Liquid Flow Rate \n',wrap=True)
plt.xlabel('Liquid Flow Rate L/min.')
plt.ylabel('(m)')
plt.legend(loc='upper right')
plt.show()

fig2.savefig('HOL.pdf')

fig3 = plt.figure(figsize=(8,6))

plt.grid(color='k',linestyle='-',linewidth=1,alpha=0.1)
#plt.axis([5,8,0,55])
plt.plot(Qdot_water_av,NOG_av,'b',marker='o',label="Empirical")
plt.errorbar(Qdot_water_av,NOG_av, yerr=NOG_err, Linestyle = "None")
plt.plot(Qdot_water_av,NOG_exp_av,'--r',marker='x',label="Experimental")
plt.errorbar(Qdot_water_av,NOG_exp_av, yerr=NOG_exp_err, Linestyle = "None")
plt.title('Number of Gas Phase Mass Transfer Units vs. Liquid Flow Rate \n',wrap=True)
plt.xlabel('Liquid Flow Rate L/min.')
plt.ylabel('# of Mass Units')
plt.legend(loc='center right')
plt.show()

fig3.savefig('NOG.pdf')

fig4 = plt.figure(figsize=(8,6))

plt.grid(color='k',linestyle='-',linewidth=1,alpha=0.1)
#plt.axis([5,8,0,4])
plt.plot(Qdot_water_av, NOL_av, 'b',marker='o',label="Empirical")
plt.errorbar(Qdot_water_av,NOL_av, yerr=NOL_err, Linestyle = "None")
plt.plot(Qdot_water_av, NOL_exp_av,'--r',marker='x',label="Experimental")
plt.errorbar(Qdot_water_av,NOL_exp_av, yerr=NOL_exp_err, Linestyle = "None")
plt.title('Number of Liquid Phase Mass Transfer Units vs. Liquid Flow Rate \n',wrap=True)
plt.xlabel('Liquid Flow Rate L/min.')
plt.ylabel('# of Mass Units')
plt.legend(loc='best')
plt.show()

fig4.savefig('NOL.pdf')     
      
fig5 = plt.figure(figsize=(8,6))

plt.grid(color='k',linestyle='-',linewidth=1,alpha=0.1)
#plt.axis([5,8,0.00001,0.009])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText = True)
plt.plot(Qdot_water_av,Ky_av,'k',marker='o',label="Empirical")
plt.errorbar(Qdot_water_av,Ky_av, yerr=Ky_err, Linestyle = "None")
plt.plot(Qdot_water_av,Ky_exp_av,'--b',marker='x',label="Experimental")
plt.errorbar(Qdot_water_av,Ky_exp_av, yerr=Ky_exp_err, Linestyle = "None")
plt.title('Overall Mass Transfer Coefficient in Gas Phase vs. Liquid Flow Rate \n',wrap=True)
plt.xlabel('Liquid Flow Rate L/min.')
plt.ylabel('mole/[(m^2*s)]')
plt.legend(loc='center right')
plt.show()

fig5.savefig('Ky.pdf')


fig6 = plt.figure(figsize=(8,6))

plt.grid(color='k',linestyle='-',linewidth=1,alpha=0.1)
#plt.axis([5,8,0.000014,0.00018])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText = True)
plt.plot(Qdot_water_av, Kx_av, 'k',marker='o',label="Empirical")
plt.errorbar(Qdot_water_av,Kx_av, yerr=Kx_err, Linestyle = "None")
plt.plot(Qdot_water_av,Kx_exp_av,'--b',marker='x',label="Experimental")
plt.errorbar(Qdot_water_av,Kx_exp_av, yerr=Kx_exp_err, Linestyle = "None")
plt.title('Overall Mass Transfer Coefficient in Liquid Phase vs. Liquid Flow Rate \n',wrap=True)
plt.xlabel('Liquid Flow Rate L/min.')
plt.ylabel('mole/[(m^2*s)]')
plt.legend(loc='center right')
plt.show()

fig6.savefig('Kx.pdf')

fig7 = plt.figure(figsize=(8,6))

plt.grid(color='k',linestyle='-',linewidth=1,alpha=0.1)
#plt.axis([5,8,0.000014,0.00018])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText = True)
plt.plot(Qdot_water_av, eta_av, 'g',marker='o')
plt.title('Column Efficiency vs Liquid Flowrate \n',wrap=True)
plt.xlabel('Liquid Flow Rate L/min.')
plt.ylabel('$CO2Absorbed (moles trans/ moles in)$')

fig6.savefig('Eta.pdf')


#Error Analysis 

Y_co2ins = (Y_co2in[0] + Y_co2in[3])/2
X_co2outs = (X_co2out[0] + X_co2out[3]) / 2 
Y_co2outs = (Y_co2out[0] + Y_co2out[3]) / 2 

L_primes = (L_prime_mol[0] + L_prime_mol[3]) / 2 
V_primes = (V_prime_mol[0] + V_prime_mol[3]) / 2 

stdyin = np.std([Y_co2in[0] , Y_co2in[3]])

stdyout = np.std([Y_co2out[0] , Y_co2out[3]])

stdxout = np.std([X_co2out[0] , X_co2out[3]])

gammayin = (12.71*stdyin) / np.sqrt(2)

gammayout = (12.71*stdyout) /np.sqrt(2)

gammaxout = (12.71*stdxout) / np.sqrt(2)

dfdyin = (V_primes*(1/(Y_co2ins - 1) - Y_co2ins/(Y_co2ins - 1)**2))/(L_primes*(X_co2outs/(X_co2outs - 1) + (V_primes*(Y_co2ins/(Y_co2ins - 1) - Y_co2outs/(Y_co2outs - 1)))/L_primes - 1)) - (V_primes*(1/(Y_co2ins - 1) - Y_co2ins/(Y_co2ins - 1)**2)*(X_co2outs/(X_co2outs - 1) + (V_primes*(Y_co2ins/(Y_co2ins - 1) - Y_co2outs/(Y_co2outs - 1)))/L_primes))/(L_primes*(X_co2outs/(X_co2outs - 1) + (V_primes*(Y_co2ins/(Y_co2ins - 1) - Y_co2outs/(Y_co2outs - 1)))/L_primes - 1)**2)

dfdyout = (V_primes*(1/(Y_co2outs - 1) - Y_co2outs/(Y_co2outs - 1)**2)*(X_co2outs/(X_co2outs - 1) + (V_primes*(Y_co2ins/(Y_co2ins - 1) - Y_co2outs/(Y_co2outs - 1)))/L_primes))/(L_primes*(X_co2outs/(X_co2outs - 1) + (V_primes*(Y_co2ins/(Y_co2ins - 1) - Y_co2outs/(Y_co2outs - 1)))/L_primes - 1)**2) - (V_primes*(1/(Y_co2outs - 1) - Y_co2outs/(Y_co2outs - 1)**2))/(L_primes*(X_co2outs/(X_co2outs - 1) + (V_primes*(Y_co2ins/(Y_co2ins - 1) - Y_co2outs/(Y_co2outs - 1)))/L_primes - 1))

dfdxout = (1/(X_co2outs - 1) - X_co2outs/(X_co2outs - 1)**2)/(X_co2outs/(X_co2outs - 1) + (V_primes*(Y_co2ins/(Y_co2ins - 1) - Y_co2outs/(Y_co2outs - 1)))/L_primes - 1) - ((1/(X_co2outs - 1) - X_co2outs/(X_co2outs - 1)**2)*(X_co2outs/(X_co2outs - 1) + (V_primes*(Y_co2ins/(Y_co2ins - 1) - Y_co2outs/(Y_co2outs - 1)))/L_primes))/(X_co2outs/(X_co2outs - 1) + (V_primes*(Y_co2ins/(Y_co2ins - 1) - Y_co2outs/(Y_co2outs - 1)))/L_primes - 1)**2

prop = np.sqrt((gammayin**2)*(dfdyin**2)   +  (gammayout**2)*(dfdyout**2)  +  (gammaxout**2)*(dfdxout**2))

#Efficiency Calculations

alpha = (L_prime_mol/V_prime_mol)*((Y_co2in/H)/(1-Y_co2in/H) - (Y_co2in)/(1-Y_co2in))
Y_co2out_theor = alpha/(1 + alpha)
percab_theor = 1 - (Y_co2out_theor/Y_co2in)*(V1/V2)

eff = eta / percab_theor

