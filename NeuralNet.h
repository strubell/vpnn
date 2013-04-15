/*
 * NeuralNet.h
 *
 *  Created on: Mar 23, 2013
 *      Author: ema
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <eigen3/Eigen/Dense>
#include <mpreal.h>

using namespace mpfr;
using namespace Eigen;

typedef Array<mpreal,Dynamic,Dynamic> MPMatrix;
typedef Array<mpreal,Dynamic,1> MPVector;

class NeuralNet {
protected:
	MPMatrix hiddenWeights;
	MPMatrix outputWeights;
	mpreal errorTol;
	int numInput;
	int numHidden;
	int numOutput;
	unsigned long currentPrecision;
public:
	NeuralNet(int numInput, int numHidden, int numOutput, unsigned long initialPrecision, mpreal errorTol);

	virtual MPMatrix train(MPMatrix &inputs, MPMatrix &desiredOutputs, MPMatrix &errors, mpreal eta, int iters, int verbose) = 0;
	virtual MPVector train(MPVector &input, MPVector &desiredOutput, MPVector &error, mpreal eta, int verbose) = 0;
	virtual MPMatrix test(MPMatrix &inputs, MPMatrix &outputs, MPMatrix &errors, int verbose) = 0;

	void setPrecision(unsigned long prec);
	MPVector sigmoid(MPVector vec);
	void printMPMatrix(MPMatrix m);

	virtual ~NeuralNet() = 0;
};

#endif /* NEURALNET_H_ */
