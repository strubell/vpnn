from numpy import *
import matplotlib.pyplot as plt
from matplotlib import rc
import subprocess

# configure/use LaTeX for plot fonts
rc('font', family='serif')
rc('text', usetex=True)

data_fname = "bin/output"
programName = "./bin/VPNNet-exp"

setSize = 5
initialPrecision = 2
trainingIters = 100
interval = 1.0
min_k1 = 1.0
max_k1 = 10.0
min_k2 = min_k1
max_k2 = max_k1
eta = 1.0

k1_vals = linspace(min_k1, max_k1, (min_k1+max_k1)/interval)
k2_vals = linspace(min_k2, max_k2, (min_k2+max_k2)/interval)
iteration = arange(trainingIters)
precisions = empty(trainingIters)
errors = empty(trainingIters)

for i,k1 in enumerate(k1_vals):
	for j,k2 in enumerate(k2_vals):
	# run experiment with constants k1, k2
	print "Running experiment with k1 = %g, k2 = %g" % (k1, k2)
	subprocess.call([programName, "-p", str(initialPrecision), "-s", str(setSize), "-i", str(trainingIters), "-l", str(eta), "-k", k1, k2, "-o", data_fname])
	numVals = 0.0
	for line in open(data_fname, 'r'):
		accuracies[idx] += float(line)
		numVals += 1.0
	accuracies[idx] /= numVals	# mean accuracy for precision prec

fig = plt.figure()
ax = fig.gca()
ax.plot(iteration, error, color="black")
#ax.set_yscale("log")
ax.set_ylabel("Training Squared Error")
ax.set_xlabel("Iteration")
ax.set_title("Error vs. (Constant) Precision (set size = %d)" % (setSize))

#plt.legend(["min = %f" % (r_min1), "min = %f" % (r_min2), "min = %f" % (r_min3)], shadow=True)
#plt.xlim((r_low,r_high-r_int))

saveFormat = "pdf"
saveName = "data/error-vs-prec-ss" + str(setSize) + "-iters" + str(trainingIters) + "-eta" + str(eta) + "." + saveFormat
plt.savefig(saveName)
plt.show()

