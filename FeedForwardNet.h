
#ifndef FEEDFORWARDNET_H_
#define FEEDFORWARDNET_H_

#include <eigen3/Eigen/Dense>
#include <mpreal.h>
#include "NeuralNet.h"

using namespace mpfr;
using namespace Eigen;

typedef Array<mpreal,Dynamic,Dynamic> MPMatrix;
typedef Array<mpreal,Dynamic,1> MPVector;

class FeedForwardNet : virtual public NeuralNet{
private:

public:
	/* Constructor: takes size of three network layers */
	FeedForwardNet(int numInput, int numHidden, int numOutput, unsigned long initialPrecision, mpreal errorTol);

	/* Trains network on given input and output data for specified
	 * number of iterations. Returns an array containing error
	 * measured at each iteration */
	MPMatrix train(MPMatrix &inputs, MPMatrix &desiredOutputs, MPMatrix &errors, MPVector &constants, mpreal eta, int iters, int squareError, int verbose);
	MPVector train(MPVector &input, MPVector &desiredOutput, MPVector &error, mpreal eta, int squareError, int verbose);


	/* Tests network on given input and output data; Returns
	 * an array containing errors */
	MPMatrix test(MPMatrix &inputs, MPMatrix &outputs, MPMatrix &errors, int verbose);

	/* Destructor */
	~FeedForwardNet();
};



#endif /* FEEDFORWARDNET_H_ */
