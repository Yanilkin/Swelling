import numpy as np
import time
from scipy.integrate import odeint

# Parameters
import parameters as p
G = p.G
max_N = p.max_N
max_n = p.max_n
max_c = p.max_c

# Material constants
import materials_constants as mc
sizes = mc.sizes
sizes_m = mc.sizes_m
Di = mc.Di
Dv =  mc.Dv
Dm = mc.Dm
k2is = mc.k2is
k2id = mc.k2id
k2vs = mc.k2vs
k2vd = mc.k2vd
k2vp = mc.k2vp # Sink strength of pores
k2mp = mc.k2mp # Sink strength of pores
k2ip = mc.k2ip # Sink strength of pores
k2vp_c = mc.k2vp_c # Sink strength of pores for rate constant 
k2mp_c = mc.k2mp_c # Sink strength of pores for rate constant 
k2ip_c = mc.k2ip_c # Sink strength of pores for rate constant
k2iv = mc.k2iv
cveq = mc.cveq
cVeq = mc.cVeq
cmeq = mc.cmeq
prbias = mc.prbias

def Dicif(k2p):
	return G/(k2is+k2id+k2p) # Calculation of SIA concentration from stationary

def f(c_all,t):
	"""Growth rate equations for pore distribution"""
	#c[0] = 1e-2
	cm = c_all[0]
	c = np.reshape(c_all[1:max_c*max_n+1],(max_c,max_n))
	cV = c_all[max_c*max_n+1:max_c*max_n+max_N-max_n+1]
	cVm = c_all[max_c*max_n+max_N-max_n+1:max_c*max_n+2*max_N-max_n+1]	
	k2v = np.sum(k2vp[:,:max_n]*c[:,:]) + np.sum(k2vp[0,max_n:max_N-1]*cV[:-1])  # Total sink strength of voids for vacancies
	k2m = np.sum(k2mp[:-1,:max_n]*c[:-1,:])  # Total sink strength of voids for vacancies
	k2i = np.sum(k2ip[:,1:max_n]*c[:,1:]) + np.sum(k2ip[0,max_n:max_N-1]*cV[:-1]) + k2ip_c[0,0]*c[0,0] + k2ip_c[1,0]*c[1,0] + k2ip_c[2,0]*c[2,0] # Total sink strength of voids for vacancies
	# I. Calculation of SIA concentration from stationary
	#Dici = Dicif(k2i)
	Dici = 0
	Dvcv = Dv*c[0][0]
	#print 'Dvcv=',Dvcv
	Dmcm = Dm*cm
	#print 'Dmcm=', Dmcm
	# II. Changes in vacancy concentration
	dcm = -k2m*Dmcm - k2mp[0,0]*Dvcv*cm + Dv*k2mp[0,0]*cmeq[1,0]*c[1,0] + Dm*np.sum(k2mp[:-1,:max_n]*cmeq[1:,:]*c[1:,:]) + k2ip_c[1,0]*Dici*c[1,0] + 2*k2ip_c[2,0]*Dici*c[2,0]
	
	dc = np.zeros((max_c,max_n))
	dc[0,0] = G*prbias[0]*0 - ((k2vs+k2vd+k2v+k2vp[0,0]*c[0,0]+k2mp[0,0]*cm)*Dv + k2ip_c[0,0]*Dici + k2mp_c[0,0]*Dmcm)*c[0,0] + k2ip_c[0,1]*Dici*c[0,1] + k2vp_c[0,0]*Dv*cveq[0,1]*c[0,1] + k2mp_c[0,0]*(Dm+Dv)*cmeq[1,0]*c[1,0] + Dv*np.sum(k2vp[:,:max_n-1]*cveq[:,1:]*c[:,1:])
	dc[0,0] = dc[0,0] + Dv*np.sum(k2vp[0,max_n-1:max_N-1]*cVeq[0:]*cV[0:])
#	dc[0][0] = 0
	# III. Changes in clusters concentration
	# a. v[j], where nj- cluster number
	i = 0
	pl = k2vp_c[i,0:max_n-2]*Dvcv*c[i,0:max_n-2] + k2ip_c[i,2:max_n]*Dici*c[i,2:max_n] + k2vp_c[i,1:max_n-1]*Dv*cveq[i,2:max_n]*c[i,2:max_n]
	m = (k2vp_c[i,1:max_n-1]*Dvcv + k2ip_c[i,1:max_n-1]*Dici + k2vp_c[i,0:max_n-2]*Dv*cveq[i,1:max_n-1])*c[i,1:max_n-1]
	plc = k2mp_c[i,1:max_n-1]*Dm*cmeq[i+1,1:max_n-1]*c[i+1,1:max_n-1]
	mc = k2mp_c[i,1:max_n-1]*Dmcm*c[i,1:max_n-1]
	dc[i,1:max_n-1] = pl-m + plc-mc
	# b. c[i]v, where i - cluster number
	j = 0
	pl = k2ip_c[1:max_c-1,j+1]*Dici*c[1:max_c-1,j+1] + k2vp_c[1:max_c-1,j]*Dv*cveq[1:max_c-1,j+1]*c[1:max_c-1,j+1]
	m = k2vp_c[1:max_c-1,j]*Dvcv*c[1:max_c-1,j]
	plc = k2mp_c[0:max_c-2,j]*Dmcm*c[0:max_c-2,j] + k2mp_c[1:max_c-1,j]*Dm*cmeq[2:max_c,j]*c[2:max_c,j]
	mc = (k2mp_c[1:max_c-1,j]*Dmcm + k2mp_c[0:max_c-2,j]*Dm*cmeq[1:max_c-1,j])*c[1:max_c-1,j]
	dc[1:max_c-1,j] = pl-m + plc-mc

	# c. c[i]v[j], where i,j - cluster number
	pl = k2vp_c[1:max_c-1,0:max_n-2]*Dvcv*c[1:max_c-1,0:max_n-2] + k2ip_c[1:max_c-1,2:max_n]*Dici*c[1:max_c-1,2:max_n] + k2vp_c[1:max_c-1,1:max_n-1]*Dv*cveq[1:max_c-1,2:max_n]*c[1:max_c-1,2:max_n]
	m = (k2vp_c[1:max_c-1,1:max_n-1]*Dvcv + k2ip_c[1:max_c-1,1:max_n-1]*Dici + k2vp_c[1:max_c-1,0:max_n-2]*Dv*cveq[1:max_c-1,1:max_n-1])*c[1:max_c-1,1:max_n-1]
	plc = k2mp_c[0:max_c-2,1:max_n-1]*Dmcm*c[0:max_c-2,1:max_n-1] + k2mp_c[1:max_c-1,1:max_n-1]*Dm*cmeq[2:max_c,1:max_n-1]*c[2:max_c,1:max_n-1]
	mc = (k2mp_c[1:max_c-1,1:max_n-1]*Dmcm + k2mp_c[0:max_c-2,1:max_n-1]*Dm*cmeq[1:max_c-1,1:max_n-1])*c[1:max_c-1,1:max_n-1]
	dc[1:max_c-1,1:max_n-1] = pl-m + plc-mc

	# d. c_maxv[j], where j - cluster number
	i = max_c-1
	pl = k2vp_c[i,0:max_n-2]*Dvcv*c[i,0:max_n-2] + k2ip_c[i,2:max_n]*Dici*c[i,2:max_n] + k2vp_c[i,1:max_n-1]*Dv*cveq[i,2:max_n]*c[i,2:max_n]
	m = (k2vp_c[i,1:max_n-1]*Dvcv + k2ip_c[i,1:max_n-1]*Dici + k2vp_c[i,0:max_n-2]*Dv*cveq[i,1:max_n-1])*c[i,1:max_n-1]
	plc = k2mp_c[i-1,1:max_n-1]*Dmcm*c[i-1,1:max_n-1] 
	mc = k2mp_c[i-1,1:max_n-1]*Dm*cmeq[i,1:max_n-1]*c[i,1:max_n-1]
	dc[i,1:max_n-1] = pl-m + plc-mc 

	# e. c[i]vmax, where i - cluster number
	j = max_n-1
	pl = k2vp_c[1:max_c-1,j-1]*Dvcv*c[1:max_c-1,j-1]
	m = (k2vp_c[1:max_c-1,j]*Dvcv + k2ip_c[1:max_c-1,j]*Dici + k2vp_c[1:max_c-1,j-1]*Dv*cveq[1:max_c-1,j])*c[1:max_c-1,j]
	plc = k2mp_c[0:max_c-2,j]*Dmcm*c[0:max_c-2,j] + k2mp_c[1:max_c-1,j]*Dm*cmeq[2:max_c,j]*c[2:max_c,j]
	mc = (k2mp_c[1:max_c-1,j]*Dmcm + k2vp_c[0:max_c-2,j]*Dm*cmeq[1:max_c-1,j])*c[1:max_c-1,j]
	dc[1:max_c-1,j] = pl-m + plc-mc

	# f. v_max, where nj- cluster number
	i = 0
	j = max_n-1
	pl = k2vp_c[i,j-1]*Dvcv*c[i,j-1]
	m = (k2vp_c[i,j]*Dvcv + k2ip_c[i,j]*Dici + k2vp_c[i,j-1]*Dv*cveq[i,j])*c[i,j]
	plc = k2mp_c[i,j]*Dm*cmeq[i+1,j]*c[i+1,j]
	mc = k2mp_c[i,j]*Dmcm*c[i,j]
	dc[i,j] = pl-m + plc-mc
	# g. c_max v, where i - cluster number
	j = 0
	i = max_c-1
	pl = k2ip_c[i,j+1]*Dici*c[i,j+1] + k2vp_c[i,j]*Dv*cveq[i,j+1]*c[i,j+1]
	m = k2vp_c[i,j]*Dvcv*c[i,j]
	plc = k2mp_c[i-1,j]*Dmcm*c[i-1,j]
	mc = k2mp_c[i-1,j]*Dm*cmeq[i,j]*c[i,j]
	dc[i,j] = pl-m + plc-mc

	# k. Last one
	j = max_n-1
	i = max_c-1
	pl = k2vp_c[i,j-1]*Dvcv*c[i,j-1]
	m = (k2vp_c[i,j]*Dvcv + k2ip_c[i,j]*Dici + k2vp_c[i,j-1]*Dv*cveq[i,j])*c[i,j]
	plc = k2mp_c[i-1,j]*Dmcm*c[i-1,j]
	mc = k2mp_c[i-1,j]*Dm*cmeq[i,j]*c[i,j]
	dc[i,j] = pl-m + plc-mc

	# b. additional interaction with SIA with carbon generation
	dc[1,0]= dc[1,0] - k2ip_c[1,0]*Dici*c[1,0] + k2mp[0,0]*Dvcv*cm - Dv*k2mp[0,0]*cmeq[1,0]*c[1,0]
	dc[2,0] = dc[2,0] - k2ip_c[2,0]*Dici*c[2,0]
#	print "dc[0,0]=", dc[0,0]
	dc[0,max_n-1] = dc[0,max_n-1] + k2vp_c[0,max_n-1]*Dv*cVeq[0]*cV[0] + k2ip_c[0,max_n]*Dici*cV[0] # Changes in the matrix
	
	dcV = np.zeros((max_N-max_n))
	dcV[0] = np.sum(k2vp_c[:,max_n-1]*Dvcv*c[:,max_n-1]) - k2vp_c[0,max_n]*Dvcv*cV[0] + (k2ip_c[0,max_n+1]*Dici + k2vp_c[0,max_n]*Dv*cVeq[1])*cV[1] - k2vp_c[0,max_n-1]*Dv*cVeq[0]*cV[0] - k2ip_c[0,max_n]*Dici*cV[0] 
	pl = k2vp_c[0,max_n:max_N-2]*Dvcv*cV[0:max_N-max_n-2] + k2ip_c[0,max_n+2:max_N]*Dici*cV[2:max_N-max_n] + k2vp_c[0,max_n+1:max_N-1]*Dv*cVeq[2:max_N-max_n]*cV[2:max_N-max_n] # 
	m = (k2vp_c[0,max_n+1:max_N-1]*Dvcv + k2ip_c[0,max_n+1:max_N-1]*Dici + k2vp_c[0,max_n:max_N-2]*Dv*cVeq[1:max_N-max_n-1])*cV[1:max_N-max_n-1] #
	dcV[1:max_N-max_n-1] = pl - m
	dcV[-1] = k2vp_c[0,max_N-2]*Dvcv*cV[max_N-max_n-2] - (k2ip_c[0,max_N-1]*Dici + k2vp_c[0,max_N-2]*Dv*cVeq[max_N-max_n-1])*cV[max_N-max_n-1] 

	dcVm = np.zeros((max_N-max_n))	
	dcVm[0] = np.sum(k2vp_c[:,max_n-1]*Dvcv*c[:,max_n-1]*sizes_m) - k2vp_c[0,max_n]*Dvcv*cVm[0] + (k2ip_c[0,max_n+1]*Dici + k2vp_c[0,max_n]*Dv*cVeq[1])*cVm[1] - k2vp_c[0,max_n-1]*Dv*cVeq[0]*cVm[0] - k2ip_c[0,max_n]*Dici*cVm[0]
	pl = k2vp_c[0,max_n:max_N-2]*Dvcv*cVm[0:max_N-max_n-2] + (k2ip_c[0,max_n:max_N-2]*Dici + k2vp_c[0,max_n+1:max_N-1]*Dv*cVeq[1:max_N-max_n-1])*cVm[1:max_N-max_n-1] 
	m =  k2vp_c[0,max_n+1:max_N-1]*Dvcv*cVm[1:max_N-max_n-1] + (k2ip_c[0,max_n+1:max_N-1]*Dici + k2vp_c[0,max_n+2:max_N]*Dv*cVeq[2:max_N-max_n])*cVm[2:max_N-max_n]
	dcVm[1:max_N-max_n-1] = pl - m
	dcVm[-1] = 

	dc = np.reshape(dc,max_c*max_n)
	dc = np.hstack((dcm,dc,dcV,dcVm))
	#print dc
	return dc

if __name__ == "__main__":

	#print "initil derivitives\n", f(p.co,0)

	# Time integration of grow equation
	t = np.linspace(0,p.max_dose/G,p.integ_steps)
	#print f(p.co,t)
	time_init = time.time()
	results = odeint(f,p.co,t)
	print "Calculation time = ", time.time()-time_init
	cm = results[:,0]
	clusters = results[:,1:max_n*max_c+1]
	clusters_v = results[:,max_n*max_c+1:max_n*max_c+max_N-max_n+1]
	concen_v = results[:,max_n*max_c+max_N-max_n+1:max_n*max_c+2*max_N-max_n+1]
	#print "Clusters\n", results[:,1:]
	#print np.shape(clusters)

	#print "final distribution\n", np.reshape(clusters[-1],(max_c,max_n))
	print "final distribution\n", np.hstack((clusters[-1,0:max_n],clusters_v[-1]))
	print "concentration in clusters\n", concen_v[-1]

	#print "total vacancy concentration\n", np.sum(np.hstack((clusters[-1,0:max_n],clusters_v[-1]))*sizes)	
	#print "total vacancy concentration\n", np.sum(clusters[-1,0:max_n]*sizes[0:max_n]) + 10*np.sum(clusters_v[-1])
	print "total vacancy concentration\n", np.sum(clusters[-1]*np.tile(sizes[:max_n],max_c))+np.sum(clusters_v[-1]*sizes[max_n:max_N])
	#print "total swelling\n," np.sum(results[-1]*sizes/Va)

	print "cluster distribution\n", np.reshape(clusters[-1],(max_c,max_n))[:,0]
	print "mobile impurity concentration\n", cm[-1]
	#print np.repeat(np.arange(1,max_c,1),max_n)
	#print clusters[-1,max_n:]
	print "total impurity concentration\n", cm[-1]+np.sum(clusters[-1,:]*np.repeat(sizes_m,max_n))+np.sum(concen_v[-1])
	#print "total swelling\n," np.sum(results[-1]*sizes/Va)

	#save results
	fo = open('log','w')
	fo.write('%i %i \n' % (max_c,max_n))
	for size in sizes:
		fo.write('%f ' % size)
	fo.write('\n')
	for dsize in mc.dsizes:
		fo.write('%f ' % dsize)
	fo.write('\n')
	i = 0
	for rows in clusters:
		if i%p.discretization ==0: 
			fo.write('%f '%(t[i]*G))
			for cols in rows:
				fo.write('%e ' % cols)
			fo.write('\n')
		i = i + 1
	fo.close()

