from numpy import *
import matplotlib.pyplot as plt
from matplotlib import rc
import subprocess

# configure/use LaTeX for plot fonts
rc('font', family='serif')
rc('text', usetex=True)

data_fname = "bin/output"
programName = "./bin/VPNNet-exp"

verbose = True			# whether or not to print running solution
TOL = finfo(float).eps 	# error tolerance (machine epsilon)

setSize = 5
minPrecision = 2
maxPrecision = 16
trainingIters = 1
eta = 1.0
numValues = maxPrecision-minPrecision

precisions = arange(minPrecision, maxPrecision+1)
accuracies = zeros(len(precisions))

#for i,line in enumerate(open(data_fname, 'r')):
#	prec[i], accuracy[i] = line.split(" ")
for idx,prec in enumerate(precisions):
	# run experiment with precision = prec >> data_fname
	print "Running experiment with precision", prec
	subprocess.call([programName, "-p", str(prec), "-s", str(setSize), "-i", str(trainingIters), "-l", str(eta), "-o", data_fname])
	numVals = 0.0
	for line in open(data_fname, 'r'):
		accuracies[idx] += float(line)
		numVals += 1.0
	accuracies[idx] /= numVals	# mean accuracy for precision prec

fig = plt.figure()
ax = fig.gca()
ax.plot(precisions, accuracies, color="black")
#ax.set_yscale("log")
ax.set_ylabel("Mean Squared Error")
ax.set_xlabel("Precision (bits)")
ax.set_title("Error vs. (Constant) Precision (set size = %d)" % (setSize))

#plt.legend(["min = %f" % (r_min1), "min = %f" % (r_min2), "min = %f" % (r_min3)], shadow=True)
#plt.xlim((r_low,r_high-r_int))

saveFormat = "pdf"
saveName = "data/error-vs-prec-ss" + str(setSize) + "-iters" + str(trainingIters) + "-eta" + str(eta) + "." + saveFormat
plt.savefig(saveName)
plt.show()

