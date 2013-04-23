from numpy import *
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.collections import LineCollection
import subprocess

# configure/use LaTeX for plot fonts
rc('font', family='serif')
rc('text', usetex=True)

data_fname = "bin/output"
programName = "./bin/VPNNet-exp"

setSize = 10
initialPrecision = 4
trainingIters = 125
min_k1 = 0.0
max_k1 = 10.0
min_k2 = min_k1
max_k2 = max_k1
interval = 1.0
eta = 5.0

vary_precision = True
vary_constants = True

if not vary_constants: 
	if vary_precision:
		# varying precision but only with one set of constants
		interval = 2.0
	else:
		# not messing with precision at all
		interval = 1.0
	
num_k1_vals = (min_k1+max_k1+1.0)/interval
num_k2_vals = (min_k2+max_k2+1.0)/interval

k1_vals = linspace(min_k1, max_k1, num_k1_vals)
k2_vals = linspace(min_k2, max_k2, num_k2_vals)
iterations = arange(trainingIters)
precisions = empty((num_k1_vals,num_k2_vals,trainingIters))
errors = empty((num_k1_vals,num_k2_vals,trainingIters))

for i,k1 in enumerate(k1_vals):
	for j,k2 in enumerate(k2_vals):
		# run experiment with constants k1, k2
		print "Running experiment with k1 = %g, k2 = %g" % (k1, k2)
		subprocess.call([programName, "-p", str(initialPrecision), "-s", str(setSize), "-i", str(trainingIters), "-l", str(eta), "-k", str(k1), str(k2), "-o", data_fname])
		numVals = 0.0
		# writes precision followed by error per line
		for k,line in enumerate(open(data_fname, 'r')):
			err, prec = line.split(" ")
			precisions[i,j,k] = long(prec)
			errors[i,j,k] = float(err)

fig = plt.figure()
ax = fig.gca()

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be numlines x points per line x 2 (x and y)
for i in range(int(num_k1_vals)):
	for j in range(int(num_k2_vals)):
		points = array([iterations, errors[i,j]]).T.reshape(-1, 1, 2)
		segments = concatenate([points[:-1], points[1:]], axis=1)
 		
		# Create the line collection object, setting the colormapping parameters.
		# Have to set the actual values used for colormapping separately.
		lc = LineCollection(segments, cmap=plt.get_cmap('rainbow'))#, norm=plt.Normalize(0, 10))
		lc.set_array(precisions[i,j])
		lc.set_linewidth(1)
 
		ax.add_collection(lc)

# for i in range(int(num_k1_vals)):
#  	for j in range(int(num_k2_vals)):
#  		ax.plot(iterations,errors[i,j])

#ax.set_yscale("log")
ax.set_ylabel("Training Error (Squared)")
ax.set_xlabel("Iteration")
ax.set_title("$\Delta$ Training Error, Precision over Time ($s$ = %d, $\eta$ = %g, $p$ = %g)" % (setSize, eta, initialPrecision))
#ax.set_ylim([amin(errors), amax(errors)])
ax.set_ylim([0.0, 1.0])
ax.set_xlim([0.0,trainingIters])

# don't bother making color bar if not changing precision
if vary_precision:
	cb = plt.colorbar(lc)
	cb.set_label("Precision (bits)")

#plt.legend(["min = %f" % (r_min1), "min = %f" % (r_min2), "min = %f" % (r_min3)], shadow=True)
#plt.xlim((r_low,r_high-r_int))

saveFormat = "pdf"
saveName = "data/error-prec-vs-time-ss" + str(setSize) + "-iters" + str(trainingIters) + "-eta" + str(eta) +  "-prec" + str(initialPrecision) + "." + saveFormat
plt.savefig(saveName)
print "Saved plot to file: %s" % (saveName)
plt.show()

