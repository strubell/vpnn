from numpy import *
import matplotlib.pyplot as plt
from matplotlib import rc
import subprocess

# configure/use LaTeX for plot fonts
rc('font', family='serif')
rc('text', usetex=True)

data_fname = "output"
programName = "./VPNN-exp"

verbose = True			# whether or not to print running solution
TOL = finfo(float).eps 	# error tolerance (machine epsilon)

minPrecision = 2
maxPrecision = 64
numValues = maxPrecision-minPrecision

precisions = arange(minPrecision, maxPrecision+1)
accuracies = zeros(numValues)

#for i,line in enumerate(open(data_fname, 'r')):
#	prec[i], accuracy[i] = line.split(" ")
for idx,prec in enumerate(precisions):
	# run experiment with precision = prec >> data_fname
	subprocess.call([programName, "-p " + str(prec), "-o " + data_fname])
	numVals = 0.0
	for line in open(data_fname, 'r'):
		accuracies[idx] += float(line)
		numVals += 1.0
	accuracies[idx] /= numVals	# mean accuracy for precision prec

fig = plt.figure()
ax = fig.gca()
ax.plot(precisions, accuracies, color="black")
ax.set_ylabel("Mean Accuracy")
ax.set_xlabel("Precision")

#plt.legend(["min = %f" % (r_min1), "min = %f" % (r_min2), "min = %f" % (r_min3)], shadow=True)
#plt.xlim((r_low,r_high-r_int))

plt.show()

