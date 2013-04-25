from numpy import *
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.collections import LineCollection
import subprocess
import os

def ensure_dir(f):
    if not os.path.exists(f):
		os.makedirs(f)

# configure/use LaTeX for plot fonts
rc('font', family='serif')
rc('text', usetex=True)

data_fname = "bin/output"
programName = "./bin/VPNNet-exp"

setSize = 10
initialPrecision = 8
trainingIters = 150
min_k1 = 1.0
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
	
num_k1_vals = (min_k1+max_k1)/interval
num_k2_vals = (min_k2+max_k2)/interval

k1_vals = linspace(min_k1, max_k1, num_k1_vals)
k2_vals = linspace(min_k2, max_k2, num_k2_vals)
iterations = arange(trainingIters)
#precisions = empty((num_k1_vals,num_k2_vals,trainingIters))
#errors = empty((num_k1_vals,num_k2_vals,trainingIters))
precisions = empty(trainingIters)
errors = empty(trainingIters)

fig = plt.figure()
ax = fig.gca()

ax.set_ylabel("Training Error (Squared)")
ax.set_xlabel("Iteration")
ax.set_xlim([0.0,trainingIters])

saveFormat = "pdf"
paramsPrefix = "add-error-prec-vs-time-ss" + str(setSize) + "-iters" + str(trainingIters) + "-eta" + str(eta) +  "-prec" + str(initialPrecision)
saveFolder = "data/" + paramsPrefix
ensure_dir(saveFolder)
cb = 0

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be numlines x points per line x 2 (x and y)
for i,k1 in enumerate(k1_vals):
	for j,k2 in enumerate(k2_vals):
		
		# run experiment with constants k1, k2
		print "Running experiment with k1 = %g, k2 = %g" % (k1, k2)
		subprocess.call([programName, "-p", str(initialPrecision), "-s", str(setSize), "-i", str(trainingIters), "-l", str(eta), "-k", str(k1), str(k2), "-o", data_fname])
		numVals = 0.0
		# writes precision followed by error per line
		for k,line in enumerate(open(data_fname, 'r')):
			err, prec = line.split(" ")
			#precisions[i,j,k] = long(prec)
			#errors[i,j,k] = float(err)
			precisions[k] = long(prec)
			errors[k] = float(err)
			
		ax.set_ylim([0.0, errors.max()])
		
		# now do the plotting
		points = array([iterations, errors]).T.reshape(-1, 1, 2)
		segments = concatenate([points[:-1], points[1:]], axis=1)
 		
		# Create the line collection object, setting the colormapping parameters.
		# Have to set the actual values used for colormapping separately.
		lc = LineCollection(segments, cmap=plt.get_cmap('rainbow'))#, norm=plt.Normalize(0, 10))
		lc.set_array(precisions)
		lc.set_linewidth(1)
 
		ax.add_collection(lc)
		ax.set_title("Training Error, Precision over Time ($s$ = %d, $\eta$ = %g, $p$ = %g, $k_1$ = %g, $k_2$ = %g)" % (setSize, eta, initialPrecision, k1, k2))
		
		# don't bother making color bar if not changing precision
		if vary_precision:
			if cb:
				fig.delaxes(fig.axes[1])
				fig.subplots_adjust(right=0.90)  #default right padding
			cb = plt.colorbar(lc)
			cb.set_label("Precision (bits)")
		
		saveName = saveFolder + "/" + paramsPrefix + "-k1" + str(k1) + "-k2" + str(k2) + "." + saveFormat
		plt.savefig(saveName)
		print "Saved plot to file: %s" % (saveName)
		ax.clear()

# for i in range(int(num_k1_vals)):
#  	for j in range(int(num_k2_vals)):
#  		ax.plot(iterations,errors[i,j])

#ax.set_yscale("log")


#ax.set_ylim([amin(errors), amax(errors)])


#plt.legend(["min = %f" % (r_min1), "min = %f" % (r_min2), "min = %f" % (r_min3)], shadow=True)
#plt.xlim((r_low,r_high-r_int))


#plt.show()

