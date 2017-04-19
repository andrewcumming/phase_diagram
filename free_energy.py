import numpy as np
import matplotlib.pyplot as plt

gamma_crit = 178.6

def df_OCP(gamma):
	if gamma<100.0:
		return -0.3683 + 0.00458*(gamma-100.0)
	if gamma>200.0:
		return 0.09384 + 0.00428*(gamma-200.0)
	return -0.003243*gamma + 1.8645*gamma**0.32301 - 1.7748*np.log(gamma) - 0.2316 + 10.84/gamma

def Smix(x1,RZ):
	x2 = 1.0-x1
	return x1*np.log(x1/(x1+x2*RZ)) + x2*np.log(x2*RZ/(x1+x2*RZ))

def f_liquid(gamma1,x1,RZ):
	# the liquid free energy is linear mixing plus Smix
	gamma2 = gamma1 * RZ**(5.0/3.0)
	return x1*df_OCP(gamma1) + (1.0-x1)*df_OCP(gamma2) + Smix(x1,RZ)

def f_solid(gamma1,x1,RZ):
	# deviation from linear mixing for the solid
	RZ1 = RZ-1.0
	xx = np.sqrt(1.0-x1)
	C = 0.05*RZ1**2 / ((1 + 0.64*RZ1)*(1 + 0.5*RZ1**2))	
	denom = 1 + 27.0*RZ1*xx*(xx-0.3)*(xx-0.7)*(xx-1.0)/(1.0+0.1*RZ1)
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
