import numpy as np
import matplotlib.pyplot as plt
import free_energy as FE
from scipy.optimize import fsolve
import time
from concurrent import futures

def do_gam(rat):
	return list(tangent_points(rat))

def tangent_points(rat):
	# set gamma
	gamma1 = FE.gamma_crit/rat

	# calculate the minimum free energy as a function of x
	x1 = np.arange(xsteps-1)/xsteps+1.0/xsteps
	fmin = np.array([min(FE.f_liquid(gamma1,x,RZ),FE.f_solid(gamma1,x,RZ)) for x in x1])

	# calculate the derivative dfdx
	eps = 0.0001
	f2 = np.array([min(FE.f_liquid(gamma1,x,RZ),FE.f_solid(gamma1,x,RZ)) for x in (x1+eps)])
	dfdx = (f2-fmin)/eps

	# we loop through different composition values, find the tangent line at each
	# point, and then test how many times the tangent line intersects the Fmin curve
	# When that number drops to zero or increases suddenly from zero, we have 
	# a tangent point
	lastcount = -1
	for this_x,this_fmin,this_dfdx in zip(x1,fmin,dfdx):
		# tangent line
		flin = this_fmin + this_dfdx*(x1-this_x)
		# calculate number of intersections
		fdiff = fmin-flin
		count = (fdiff<0.0).sum()
		# did this change to or from zero? 
		if lastcount>=0:
			if (lastcount ==0 and count >0) or (lastcount>0 and count ==0):
				# we've found a tangent point
				if FE.f_liquid(gamma1,this_x,RZ)<FE.f_solid(gamma1,this_x,RZ):
					point_type = -1
				else:
					point_type = 1
				yield (1.0-this_x,rat,point_type)				
		lastcount = count

t0 = time.time()

# We need to specify the charges of the two species (note: the charge ratio is what matters)
# and the range of gamma to search over:
#Z1,Z2,G1,G2 = 26,34,0.9,1.7
#Z1,Z2,G1,G2 = 8,34,0.7,12.0
#Z1,Z2,G1,G2 = 3,4,0.9,1.7
#Z1,Z2,G1,G2 = 2,3,0.8,2.0
Z1,Z2,G1,G2 = 3,5,0.7,2.4
#Z1,Z2,G1,G2 = 3,13,0.2,12.0

# charge ratio
RZ = 1.0*Z2/Z1

# number of steps in x and gamma
xsteps = 1000
gsteps = 1000

# scan through gamma and find the tangent points
rat_vec = np.arange(gsteps+1)*(G2-G1)/gsteps + G1
with futures.ProcessPoolExecutor() as executor:
	results = executor.map(do_gam, rat_vec)
results = np.array([x for res in results for x in res])
x_points = results[:,0]
gamma_points = results[:,1]
point_type = results[:,2]

# check the execution time
print("Time taken=",time.time()-t0)

# plot phase diagram
plt.scatter(x_points[point_type>0],gamma_points[point_type>0],s=4)
plt.scatter(x_points[point_type<0],gamma_points[point_type<0],s=4)
# plot <Z^5/3> gamma_crit
x2 = np.arange(199)*0.005+0.005
plt.plot(x2, ((1.0-x2) + RZ**(5.0/3.0)*x2),'k--',alpha=0.2)
plt.xlim((0.0,1.0))
plt.xlabel(r'$x_2$')
plt.ylabel(r'$\Gamma_{\rm crit}/\Gamma_1$')
plt.annotate(r'$R_Z=%d/%d$' % (Z2,Z1),xy=(0.2,0.9*(max(gamma_points)-min(gamma_points))+min(gamma_points)))
plt.savefig('phase_diagram_%d_%d.pdf' % (Z2,Z1))
