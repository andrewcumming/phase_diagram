import numpy as np
import matplotlib.pyplot as plt
import free_energy as FE
from scipy.optimize import fsolve

# We need to specify the charges of the two species (note: the charge ratio is what matters)
# and the range of gamma to search over:
#Z1,Z2,G1,G2 = 26,34,0.9,1.7
#Z1,Z2,G1,G2 = 8,34,0.7,12.0
#Z1,Z2,G1,G2 = 3,4,0.9,1.7
#Z1,Z2,G1,G2 = 2,3,0.8,2.0
#Z1,Z2,G1,G2 = 3,5,0.7,2.4
Z1,Z2,G1,G2 = 3,13,0.2,12.0

# charge ratio
RZ = 1.0*Z2/Z1

# arrays to store the tangent points
x_points = np.array([])
gamma_points = np.array([])
point_type = np.array([])

# scan through the range of gamma values
for rat in np.arange(100)*0.01*(G2-G1) + G1:

	# set gamma
	gamma1 = FE.gamma_crit/rat

	# calculate the minimum free energy as a function of x
	x1 = np.arange(199)*0.005+0.005
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
				# we've found a tangent point, so store the data point
				x_points = np.append(x_points,1.0-this_x)
				gamma_points = np.append(gamma_points,rat)
				# make a note of whether it is liquid or solid				
				if FE.f_liquid(gamma1,this_x,RZ)<FE.f_solid(gamma1,this_x,RZ):
					point_type = np.append(point_type,-1)
				else:
					point_type = np.append(point_type,1)
				#plt.plot(x1,flin)
				##plt.plot(x1,fdiff)
				#plt.show()
		lastcount = count

# plot phase diagram
plt.scatter(x_points[point_type>0],gamma_points[point_type>0],s=4)
plt.scatter(x_points[point_type<0],gamma_points[point_type<0],s=4)
plt.xlim((0.0,1.0))
plt.xlabel(r'$x_2$')
plt.ylabel(r'$\Gamma_{\rm crit}/\Gamma_1$')
plt.annotate(r'$R_Z=%d/%d$' % (Z2,Z1),xy=(0.2,0.9*(max(gamma_points)-min(gamma_points))+min(gamma_points)))
plt.savefig('phase_diagram_%d_%d.pdf' % (Z2,Z1))
