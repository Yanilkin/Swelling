import numpy as np
import parameters as p

#I. Temperature independent coefficients
a = 3.14e-10 # lattice constant in m
b = np.sqrt(3)/2*a # Burger's vector sqrt(3)/2*a
Va = (a*a*a)/2.0 # Atomic volume
pi = 3.14 # Pui number

#k2s = 1.2e13*0 # Surface sink
k2s = 5e13*0 # Surface sink
k2d = 5e13*0 # Dislocation network sink
zid = 3.0 # Bias factor
zvd = 1.0 # Bias factor
k2is = k2s
k2vs = k2s
k2id = zid*k2d
k2vd = zvd*k2d

k2iv = 4*3.14*(3.1e-10+2.9e-10)/Va # Recombination rate constant


# sink strength of vacancies and pores
max_N = p.max_N
max_n = p.max_n
max_c = p.max_c
dn1 = p.dn1 
base = p.base
# Clusters initialization
sizes = np.arange(1,dn1)
for i in range(dn1,max_N+1):
	sizes=np.hstack((sizes,(2-base)*i+(base-1)*dn1-1+np.power(base,i-dn1)))
dsizes = sizes[1:]-sizes[:-1]
#print sizes
#print dsizes

radius = np.power(sizes*3*Va/4/pi,1.0/3.0) + a/2*(3.0)**(0.5)/2 # Cluster radius

k2vp = [k2iv]
for i in range(1,max_N):
	k2vp.append(4*pi/Va*(radius[i]+radius[0])) # Last one is the correction of rate constant
k2vp_c=np.hstack((k2vp[:-1]/dsizes,k2vp[-1]/dsizes[-1]))
k2vp = np.tile(k2vp,max_c).reshape(max_c,max_N)
k2vp_c = np.tile(k2vp_c,max_c).reshape(max_c,max_N)
#print k2vp

k2mp = k2vp
k2mp_c= k2mp

#k2vp_c[:,0] = 0

k2ip = [k2iv]
for i in range(1,max_N):
	k2ip.append(4*pi/Va*(radius[i]+radius[0])) # Last one is the correction of rate constant
k2ip_c = np.hstack((k2iv,k2ip[1:]/dsizes))
k2ip = np.tile(k2ip,max_c).reshape(max_c,max_N)
k2ip_c = np.tile(k2ip_c,max_c).reshape(max_c,max_N)

for i in range(max_c):
	for j in range(max_N):
		if (i-3 > 3*(j+1)):
			k2mp_c[i,j] = 0
			k2mp[i,j] = 0

for i in range(max_c):
	for j in range(1,max_N):
		if (i-3 > 3*j):
			k2ip_c[i,j] = 0
			k2ip[i,j] = 0

for i in range(max_c):
	for j in range(max_N):
		if (i-3 > 3*(j+1)):
			k2vp_c[i,j] = 0
			k2vp[i,j] = 0


#print k2ip_c

#II. Temperature dependent coefficients
T = p.T
# Diffusion coefficients
Di = 1e-8*np.exp(-0.4*11600/T)
Dv = 3.7e-6*np.exp(-1.54*11600/T) #From Ivan Novoselov
Dm = 5.2e-6*np.exp(-163*1000/8.31/T) #From artcile
#Dm = 0
#Dv = 0
# Equlibrium concentration for dissociation rate
#from Novoselov
Ef = 3.0 #eV
a = 5.4387 # From MD Ivan Novoselov
b = 2.9587 # From MD Ivan Novoselov
# Calculation of binding and dissociation energies 
Eb_c = np.array([2.0,2.0,1.67,0.5,-10])
Ediss = np.zeros((max_c,max_n))
for j in range(max_n):
	Eb2 = np.repeat(Eb_c,sizes[j])
	for i in range(max_c):
		Ediss[i,j] = np.sum(Eb2[:i])

#print Ediss
dG = -sizes[:max_n]*Ef+a*(np.power(sizes[:max_n],0.6667)-1)+Ef - Ediss
dG[:,1:] = dG[:,1:]-0.4 # To fix binding energy of divacancy - 0.2 eV
#print dG

Eb_v = dG[:,:-1] - dG[:,1:]
Eb_m = dG[:-1,:] - dG[1:,:]
#print Eb_v
cveq = np.zeros((max_c,max_n))
cveq[:,1:] = np.exp(-Eb_v*11600/T)

cVeq = np.zeros((max_N-max_n))
dGV = -sizes[max_n-1:max_N]*Ef+a*(np.power(sizes[max_n-1:max_N],0.6667)-1)+Ef
Eb_V = dGV[:-1] - dGV[1:]
cVeq =np.exp(-Eb_V*11600/T)
cveq = np.clip(cveq,1e-10,1)
cVeq = np.clip(cVeq,1e-10,1)
#cveq[0] = np.exp(-Ef*11600/T)
#cveq[1] = np.exp(-0.2*11600/T) #Binding energy for divacancy
#Eb_v = dG[:,:-1] - dG[:,1:]
#print cveq
#print cVeq
cveq = cveq*1e-20
cVeq = cVeq*1e-20

# Equlibrium concentration of impurities 
cmeq = np.zeros((max_c,max_n))
cmeq[1:,:] = np.exp(-Eb_m*11600/T)
cmeq = np.clip(cmeq,1e-10,1)
#Eb = np.array([1.93,2.01,1.67,0.5])
#cmeq[1] = np.exp(-Eb[0]*11600/T)
#cmeq[2] = np.exp(-Eb[1]*11600/T)
#cmeq[3] = np.exp(-Eb[2]*11600/T)
#cmeq[4] = np.exp(-Eb[3]*11600/T)
#cmeq = np.tile(cmeq,max_n).reshape(max_n,max_c)
#cmeq = cmeq.T
#print cmeq
#cmeq = cmeq*1e-20

#Production bias
prbias = np.zeros(max_N)
#prbias = np.hstack((np.array([1.55,0.227,0.0688,0.0414,0.0236,0.0119,0.005]),prbias))
#prbias = np.hstack((np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0]),prbias))
prbias[0] = 1.0
prbias = prbias/np.sum(prbias*sizes)