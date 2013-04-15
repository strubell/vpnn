/*
 * FFNN.cpp
 *
 *	A feed-forward neural network with variable-precision
 *	weights.
 *
 * Author: Emma Strubell
 */

#include <iostream>
#include "FeedForwardNet.h"

using namespace std;

FeedForwardNet::FeedForwardNet(int numInput, int numHidden, int numOutput, unsigned long initialPrecision, mpreal errorTol)
: NeuralNet(numInput, numHidden, numOutput, initialPrecision, errorTol){
	this->hiddenWeights = MPMatrix::Random(numHidden, numInput);
	this->outputWeights = MPMatrix::Random(numOutput, numHidden);
}

/* Trains network on given input and output data for specified
 * number of iterations. Returns an array containing error
 * measured at each iteration */
MPMatrix FeedForwardNet::train(MPMatrix &inputs, MPMatrix &desiredOutputs, MPMatrix &errors, mpreal eta, int iters, int verbose){
	int i, r;
	MPVector input(this->numInput);
	MPVector desiredOut(this->numOutput);
	MPVector error(this->numOutput);

	for(i = 0; i < iters; ++i){

		r = rand() % inputs.rows();	// select a random input to train on

		input = inputs.row(r);
		desiredOut = desiredOutputs.row(r);
		error = errors.row(i);
		errors.row(i) << train(input, desiredOut, error, eta, verbose).transpose();
	}
	return errors;
}

MPVector FeedForwardNet::train(MPVector &input, MPVector &desiredOutput, MPVector &error, mpreal eta, int verbose){
	int i;
	MPVector hidOuts(this->numOutput);
	MPVector outputs(this->numOutput);
	MPVector outErrors(this->numOutput);
	MPVector outDelta(this->numOutput);
	MPVector hidDelta(this->numHidden);

	/* Forward propagation */
	hidOuts = (this->hiddenWeights.rowwise()*input.transpose()).rowwise().sum();
	hidOuts = sigmoid(hidOuts);

	outputs = this->outputWeights.matrix()*hidOuts.matrix();
	outputs = sigmoid(outputs);

	/* Determine errors and deltas */
	outErrors = desiredOutput - outputs;
	outDelta = outErrors*(outputs*(1.0-outputs));
	hidDelta = (hidOuts*(1.0-hidOuts));
	hidDelta *= (this->outputWeights.matrix().transpose()*outDelta.matrix()).array();

	/* Update weights */
	outputWeights += eta*(hidOuts.matrix()*outDelta.matrix()).transpose().array();
	hiddenWeights += eta*(hidDelta.matrix()*input.transpose().matrix()).array();

	/* Record squared error */
	for(i = 0; i < this->numOutput; i++){
		error(i) = outErrors(i)*outErrors(i);
	}
	return error;
}

/* Tests network on given input and output data; Returns
 * an array containing errors */
MPMatrix FeedForwardNet::test(MPMatrix &inputs, MPMatrix &outputs, MPMatrix &errors, int verbose){
	int i;
	mpreal accuracy = 0.0;
	MPVector hidOuts(this->numOutput);
	MPVector outs(this->numOutput);
	MPVector outErrors(this->numOutput);
	MPVector input(this->numInput);
	MPVector desiredOut(this->numOutput);
	for(i = 0; i < inputs.rows(); ++i){

		input = inputs.row(i);
		desiredOut = outputs.row(i);

		/* Forward propagation */
		hidOuts = (this->hiddenWeights.rowwise()*input.transpose()).rowwise().sum();
		hidOuts = sigmoid(hidOuts);
		outs = this->outputWeights.matrix()*hidOuts.matrix();
		outs = sigmoid(outs);

		/* Determine errors */
		outErrors = desiredOut - outs;

		if(verbose){
			cout << "Inputs: " << endl;
			printMPMatrix(input.transpose());

			cout << "Outputs: " << endl;
			printMPMatrix(outs.transpose());

			cout << "Desired: " << endl;
			printMPMatrix(desiredOut.transpose());
		}

		/* Record squared error */
		errors(i,0) = outErrors.matrix().dot(outErrors.matrix())/outErrors.rows();
		if(verbose)
			cout << "Mean squared error: " << errors(i,0) << "\n\n";

		if(errors(i,0) < this->errorTol)
			accuracy++;
	}
	if(verbose)
		cout << "Overall accuracy: " << (accuracy/inputs.rows())*100 << "%\n";
	return errors;
}

/* Destructor */
FeedForwardNet::~FeedForwardNet() {
	// TODO nothing to do here?
}

