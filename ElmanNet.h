/*
 * ElmanNet.h
 *
 *  Created on: Mar 23, 2013
 *      Author: ema
 */

#ifndef ELMANNET_H_
#define ELMANNET_H_

#include <eigen3/Eigen/Dense>
#include <mpreal.h>
#include "NeuralNet.h"

using namespace mpfr;
using namespace Eigen;

typedef Array<mpreal,Dynamic,Dynamic> MPMatrix;
typedef Array<mpreal,Dynamic,1> MPVector;

class ElmanNet : public NeuralNet{
private:
	MPMatrix hiddenLastDelta;

public:
	ElmanNet(int numInput, int numHidden, int numOutput, unsigned long initialPrecision, mpreal errorTol);

	MPMatrix train(MPMatrix &inputs, MPMatrix &desiredOutputs, MPMatrix &errors, MPVector &constants, mpreal eta, int iters, int squareError, int verbose);
	MPVector train(MPVector &input, MPVector &desiredOutput, MPVector &error, mpreal eta, int squareError, int verbose);

	MPVector &MPRealToUnary(mpreal val, MPVector &arr);

	/* Tests network on given input and output data; Returns
	 * an array containing errors */
	MPMatrix test(MPMatrix &inputs, MPMatrix &outputs, MPMatrix &errors, int verbose);

	~ElmanNet();
};

#endif /* ELMANNET_H_ */
