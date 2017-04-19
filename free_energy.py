import numpy as np
import matplotlib.pyplot as plt

gamma_crit = 178.6

def df_OCP(gamma):
	# the difference between liquid and solid free energies for a 
	# one-component plasma. Implements equation (9) of Medin & Cumming (2010)
	# Note that some significant digits were dropped in the paper, and have
	# been added back here to ensure the free energy is continuous at Gamma=100 and 200
	if gamma<100.0:
		return -0.3683 + 0.00458*(gamma-100.0)
	if gamma>200.0:
		return 0.09384 + 0.00428*(gamma-200.0)
	return -0.003243*gamma + 1.8645*gamma**0.32301 - 1.7748*np.log(gamma) - 0.2316 + 10.84/gamma

def Smix(x1,RZ):
	# Entropy of mixing
	x2 = 1.0-x1
	return x1*np.log(x1/(x1+x2*RZ)) + x2*np.log(x2*RZ/(x1+x2*RZ))

def f_liquid(gamma1,x1,RZ):
	# Returns the liquid free energy (minus the solid linear mixing terms)
	gamma2 = gamma1 * RZ**(5.0/3.0)
	# the liquid free energy is linear mixing plus Smix
	return x1*df_OCP(gamma1) + (1.0-x1)*df_OCP(gamma2) + Smix(x1,RZ)

def f_solid(gamma1,x1,RZ):
	# Returns the solid free energy (minus the linear mixing terms)
	# First calculate the deviation from linear mixing for the solid
	# following Ogata et al. 1993
	RZ1 = RZ-1.0
	xx = np.sqrt(1.0-x1)
	C = 0.05*RZ1**2 / ((1 + 0.64*RZ1)*(1 + 0.5*RZ1**2))	
	denom = 1 + 27.0*RZ1*xx*(xx-0.3)*(xx-0.7)*(xx-1.0)/(1.0+0.1*RZ1)
	# add the deviation from linear mixing and Smix to get the free energy
	return gamma1*x1*(1.0-x1)*C/denom + Smix(x1,RZ)


if __name__ == '__main__':	
	# Reproduce the free energy curves shown in Medin & Cumming (2010) Fig.1
	gamma_crit = 178.6
	RZ = 34.0/8.0
	gamma1 = gamma_crit/6.0

	x2 = np.arange(99)*0.01 + 0.01
	fL = np.array([f_liquid(gamma1,1.0-x,RZ) for x in x2])
	fS = np.array([f_solid(gamma1,1.0-x,RZ) for x in x2])

	plt.plot(x2,fL)
	plt.plot(x2,fS)
	plt.plot(x2,np.minimum(fS,fL))
	plt.xlabel(r'$x_2$')
	plt.ylabel(r'$\mathrm{Free\ energy}$')
	plt.show()
