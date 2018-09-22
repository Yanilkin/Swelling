#Parameters
G = 5e-4
max_dose = 1e2
T = 700

# size discretization
max_N = 20
max_n = 10
max_c = 5
dn1 = 20
base = 1.05

# integration and output
integ_steps = 10
discretization = integ_steps/10

# initial distribution
import materials_constants as mc
import numpy as np
co = np.zeros((max_c,max_n))
coV = np.zeros((max_N-max_n))
#co[0,0] = np.sqrt(G/mc.k2iv/mc.Di)
co[0,0] = 1e-2
#co[0,0] = np.exp(-3.11*11600/T)
#co[2,0] = 1e-4
#co[2,0] = 1e0
#co[3,0] = 1e0
#co[4,0] = 1e0
co = np.reshape(co,max_c*max_n)
cm = 0
co = np.hstack((cm,co,coV))
##co[133] = 1e-4

#k2i = mc.k2is + mc.k2id + np.sum(mc.k2ip[1:-1]*co[1:-1])
#k2v = mc.k2vs + mc.k2vd + np.sum(mc.k2vp[1:-1]*co[1:-1])
#co[0] = np.sqrt(np.power(k2i*mc.Di/(2*mc.k2iv*(mc.Di+mc.Dv)),2)+G*k2i*mc.Di/(mc.k2iv*(mc.Di+mc.Dv)*k2v*mc.Dv))-k2i*mc.Di/(2*mc.k2iv*(mc.Di+mc.Dv))
#print mc.Dv*co[0]
#print mc.Di*(np.sqrt(np.power(k2v*mc.Dv/(2*mc.k2iv*(mc.Di+mc.Dv)),2)+G*k2v*mc.Dv/(mc.k2iv*(mc.Di+mc.Dv)*k2i*mc.Di))-k2v*mc.Dv/(2*mc.k2iv*(mc.Di+mc.Dv)))

