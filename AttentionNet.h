/*
 * AttentionNet.h
 *
 *  Created on: Apr 1, 2013
 *      Author: ema
 */

#ifndef ATTENTIONNET_H_
#define ATTENTIONNET_H_

#include <eigen3/Eigen/Dense>
#include <mpreal.h>
#include "ElmanNet.h"
#include "NeuralNet.h"

using namespace mpfr;
using namespace Eigen;

typedef Array<mpreal,Dynamic,Dynamic> MPMatrix;
typedef Array<mpreal,Dynamic,1> MPVector;

class AttentionNet: public ElmanNet{
private:
	mpreal desiredAccuracy;
	MPVector accuracies;
	int accuraciesIdx;
public:
	AttentionNet(int numInput, int numHidden, int numOutput, unsigned long initialPrecision, mpreal errorTol, mpreal desiredAccuracy, int accuracyMemLen);

	MPMatrix train(MPMatrix &inputs, MPMatrix &desiredOutputs, MPMatrix &errors, MPVector &constants, mpreal eta, int iters, int squareError, int verbose);
	MPVector train(MPVector &input, MPVector &desiredOutput, MPVector &error, mpreal eta, int squareError, int verbose);

	/* Tests network on given input and output data; Returns
	 * an array containing errors */
	MPMatrix test(MPMatrix &inputs, MPMatrix &outputs, MPMatrix &errors, int verbose);

	/* Test on a single input/output pair */
	mpreal test(MPVector &input, MPVector &output, mpreal error, int verbose);

	void recordAccuracy(MPVector &in, MPVector &out, int verbose);
	mpreal currentAccuracy();

	~AttentionNet();
};

#endif /* ATTENTIONNET_H_ */
