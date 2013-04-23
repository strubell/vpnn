/*
 * AttentionNet.cpp
 *
 *  Created on: Apr 1, 2013
 *      Author: ema
 */

#include <iostream>
#include "AttentionNet.h"

using namespace std;

AttentionNet::AttentionNet(int numInput, int numHidden, int numOutput, unsigned long initialPrecision, mpreal errorTol, mpreal desiredAccuracy, int accuracyMemLen)
	: ElmanNet(numInput, numHidden, numOutput, initialPrecision, errorTol){
	this->desiredAccuracy = desiredAccuracy;
	this->accuracies(accuracyMemLen);
	this->accuraciesIdx = 0;
}

MPMatrix AttentionNet::train(MPMatrix &inputs, MPMatrix &desiredOutputs, MPMatrix &errors, MPVector &constants, mpreal eta, int iters, int squareError, int verbose){
	int i, r;
	mpreal k1, k2, errval, direction;
	MPVector input, output, error;
	k1 = constants(0);
	k2 = constants(1); // will probably want to comment this out if doing additive
	for(i = 0; i < iters; ++i){
		r = rand() % inputs.rows();	// select a random input to train on

		/* Record squared error */
		input = MPVector(inputs.row(r));
		output = MPVector(desiredOutputs.row(r));
		error = MPVector(errors.row(i));
		errors.row(i) << train(input, output, error, eta, squareError, verbose).transpose();
		/*if(verbose){
			cout << "Error: ";
			printMPMatrix(errors.row(i));
			cout << endl;
		}*/

		/* Grab current error */
		errval = errors(i,0);
		
		/* Record accuracy for this input/output val */
		this->recordAccuracy(input, output);

		// whether to increase or decrease precision; increase if accuracy
		// above desired, decrease if below
		// assumes squared (or otherwise always-positive) error
		direction = this->currentAccuracy() > desiredAccuracy? -1.0 : 1.0;

		// TODO un-hard-code this! 
		// (pass a pointer to a function that takes an MPVector of constants, 
		// 		returns unsigned long = new precision value)
		
		/* multiplicative */
		setPrecision(k1*direction*errval*this->currentPrecision + k2);
		
		/* additive */
		//setPrecision(this->currentPrecision + k1*direction*errval);
		
		if(verbose){
			cout << "Set network precision to " << this->currentPrecision << endl;
		}
		//cout << this->currentPrecision << " " << errval << endl;
		
		// TODO clean this up -- also throw new precision in with error return value
		errors(i,1) = this->currentPrecision;
	}
	return errors;
}

MPVector AttentionNet::train(MPVector &input, MPVector &desiredOutput, MPVector &error, mpreal eta, int squareError, int verbose){
	// inner training iteration is the same as Elman net
	return ElmanNet::train(input, desiredOutput, error, eta, squareError, verbose);
}

MPMatrix AttentionNet::test(MPMatrix &inputs, MPMatrix &outputs, MPMatrix &errors, int verbose){
	return ElmanNet::test(inputs, outputs, errors, verbose);
}

MPVector AttentionNet::test(MPVector &input, MPMatrix &output, MPVector &error){
	return ElmanNet::test(input, output, error);
}

void AttentionNet::recordAccuracy(MPVector &in, MPVector &out){
	MPVector err(out.size());

	/* Compute accuracy of net on given input/output pair */
	this->test(in, out, err);

	/* Save as binary value based on errorTol */
	accuracies(accuraciesIdx) = err < this->errorTol? 1.0: 0.0;

	/* Increment index into accuracies vector */
	accuraciesIdx = (accuraciesIdx+1) % accuracies.size();
}

mpreal AttentionNet::currentAccuracy(){
	if((accuracies > 0).all()){
		/* Compute accuracy of last accuracyMemLen iterations */
		return accuracies.sum()/accuracies.size();
	}
	/* Haven't performed enough iterations yet */
	return -1;
}

AttentionNet::~AttentionNet() {
	// TODO Auto-generated destructor stub
}

