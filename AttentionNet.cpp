/*
 * AttentionNet.cpp
 *
 *  Created on: Apr 1, 2013
 *      Author: ema
 */

#include <iostream>
#include "AttentionNet.h"

using namespace std;

AttentionNet::AttentionNet(int numInput, int numHidden, int numOutput, unsigned long initialPrecision, mpreal errorTol, mpreal desiredAccuracy)
	: ElmanNet(numInput, numHidden, numOutput, initialPrecision, errorTol){
	this->desiredAccuracy = desiredAccuracy;
}

MPMatrix AttentionNet::train(MPMatrix &inputs, MPMatrix &desiredOutputs, MPMatrix &errors, mpreal eta, int iters, int verbose){
	int i, r;
	MPVector input, output, error;
	for(i = 0; i < iters; ++i){
		r = rand() % inputs.rows();	// select a random input to train on

		/* Record squared error */
		input = MPVector(inputs.row(r));
		output = MPVector(desiredOutputs.row(r));
		error = MPVector(errors.row(i));
		errors.row(i) << train(input, output, error, eta, verbose).transpose();
	}
	return errors;
}

MPVector AttentionNet::train(MPVector &input, MPVector &desiredOutput, MPVector &error, mpreal eta, int verbose){
	// inner training iteration is the same as Elman net
	return ElmanNet::train(input, desiredOutput, error, eta, verbose);
}

MPMatrix AttentionNet::test(MPMatrix &inputs, MPMatrix &outputs, MPMatrix &errors, int verbose){
	return ElmanNet::test(inputs, outputs, errors, verbose);
}

AttentionNet::~AttentionNet() {
	// TODO Auto-generated destructor stub
}

