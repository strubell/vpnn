/*
 * ElmanNet.cpp
 *
 *  Created on: Mar 23, 2013
 *      Author: ema
 */

#include <iostream>
#include "ElmanNet.h"

using namespace std;

ElmanNet::ElmanNet(int numInput, int numHidden, int numOutput, unsigned long initialPrecision, mpreal errorTol)
	: NeuralNet(numInput, numHidden, numOutput, initialPrecision, errorTol){

	this->hiddenWeights = MPMatrix::Random(this->numHidden, this->numInput+this->numHidden);
	this->outputWeights = MPMatrix::Random(this->numOutput, this->numHidden);
}

/* Trains network on given input and output data for specified
 * number of iterations. Returns an array containing error
 * measured at each iteration */
MPMatrix ElmanNet::train(MPMatrix &inputs, MPMatrix &desiredOutputs, MPMatrix &errors, MPVector &constants, mpreal eta, int iters, int squareError, int verbose){
	int i, r;
	MPVector input, output, error;
	for(i = 0; i < iters; ++i){
		r = rand() % inputs.rows();	// select a random input to train on

		/* Record squared error */
		input = MPVector(inputs.row(r));
		output = MPVector(desiredOutputs.row(r));
		error = MPVector(errors.row(i));
		errors.row(i) << train(input, output, error, eta, squareError, verbose).transpose();
	}
	return errors;
}

MPVector ElmanNet::train(MPVector &inputVal, MPVector &desiredOutput, MPVector &error, mpreal eta, int squareError, int verbose){
	//int j;
	MPVector hidOuts(this->numOutput);
	MPVector outputs(this->numOutput);
	MPVector outErrors(this->numOutput);
	MPVector outDelta(this->numOutput);
	MPVector hidDelta(this->numHidden);
	MPVector input(this->numInput + this->numHidden);
	MPVector desiredOut(this->numOutput);
	MPVector contextUnits(this->numHidden);
	MPVector unaryInput(this->numInput);
	MPVector unaryOutput(this->numOutput);

	// reset outputs of context units
	contextUnits = MPVector::Constant(this->numHidden, 0.5);

	//for(j = 0; j < inputVal.rows(); ++j){

		/* Convert input / desired output to unary vectors */
		// TODO make this consistent with ElmanNet test code (i.e.
		// do this conversion outside this function!)
		//cout << "Converting input/output to unary vectors" << endl;
		//input << contextUnits, MPRealToUnary(inputVal(j), unaryInput);
		//desiredOut << MPRealToUnary(desiredOutput(j), unaryOutput);
		input << contextUnits, inputVal;
		desiredOut << desiredOutput;

		/* Forward propagation */
		//cout << "Forward propagation" << endl;
		hidOuts = (this->hiddenWeights.rowwise()*input.transpose()).rowwise().sum();
		hidOuts = sigmoid(hidOuts);
		outputs = this->outputWeights.matrix()*hidOuts.matrix();
		outputs = sigmoid(outputs);

		/* Determine errors and deltas */
		//cout << "Determining errors, deltas" << endl;
		outErrors = desiredOut - outputs;
		outDelta = outErrors*(outputs*(1.0-outputs));
		hidDelta = (hidOuts*(1.0-hidOuts));
		hidDelta *= (this->outputWeights.matrix().transpose()*outDelta.matrix()).array();

		/* Update weights */
		//cout << "Updating weights" << endl;
		outputWeights += eta*(hidOuts.matrix()*outDelta.transpose().matrix()).transpose().array();
		hiddenWeights += eta*(hidDelta.matrix()*input.transpose().matrix()).array();

		contextUnits = hidOuts;

		/* Record squared error */
		//cout << "Recording error" << endl;
		if(squareError)
			error << outErrors.matrix().dot(outErrors.matrix()), 0;
		else
			error << outErrors, 0;
	//}
	return error;
}

/* Tests network on given input and output data; Returns
 * an array containing errors */
MPMatrix ElmanNet::test(MPMatrix &inputs, MPMatrix &outputs, MPMatrix &errors, int verbose){
	int i;//, j;
	mpreal accuracy = 0.0;
	MPVector hidOuts(this->numOutput);
	MPVector outs(this->numOutput);
	MPVector outErrors(this->numOutput);
	MPVector input(this->numInput + this->numHidden);
	MPVector desiredOut(this->numOutput);
	MPVector contextUnits(this->numHidden);
	MPVector unaryInput(this->numInput);
	MPVector unaryOutput(this->numOutput);
	MPVector::Index maxIdx;
	for(i = 0; i < inputs.rows(); ++i){
		contextUnits = MPVector::Constant(this->numHidden, 0.5);
		//for(j = 0; j < inputs.cols(); ++j){
			/* Convert input / desired output to unary vectors */

			//TODO make this consistent with ElmanNet test in main
			//input << contextUnits, MPRealToUnary(inputs(i,j), unaryInput);
			//desiredOut << MPRealToUnary(outputs(i,j), unaryOutput);
			MPVector temp = inputs.row(i);
			input << contextUnits, temp;
			desiredOut << outputs.row(i);

			/* Forward propagation */
			//cout << "Forward propagation" << endl;
			hidOuts = (this->hiddenWeights.rowwise()*input.transpose()).rowwise().sum();
			hidOuts = sigmoid(hidOuts);
			outs = this->outputWeights.matrix()*hidOuts.matrix();
			outs = sigmoid(outs);

			/* Determine errors */
			outErrors = desiredOut - outs;

			/* Update context units */
			contextUnits = hidOuts;

			if(verbose){
				cout << "Inputs: " << endl;
				printMPMatrix(temp.transpose());

				cout << "Outputs: " << endl;
				printMPMatrix(outs.transpose());

				cout << "Desired: " << endl;
				printMPMatrix(desiredOut.transpose());
			}

			/* Record mean squared error */
			errors(i) = outErrors.matrix().dot(outErrors.matrix())/outErrors.rows();

			outs.maxCoeff(&maxIdx);
			if(maxIdx == outputs(i))
				accuracy++;

			if(verbose)
				cout << "Mean squared error: " << errors(i) << "\n\n";
		//}
	}
	if(verbose)
		cout << "Overall accuracy: " << (accuracy/(inputs.cols()*inputs.rows()))*100 << "%\n";
	return errors;
}

/* Test on a single input/output pair */
mpreal ElmanNet::test(MPVector &input, MPVector &output, mpreal error, int verbose){
	
	MPVector contextUnits(this->numHidden);
	MPVector hidOuts(this->numOutput);
	MPVector outs(this->numOutput);
	MPVector outErrors(this->numOutput);
	MPVector conInput(this->numInput + this->numHidden);

	contextUnits = MPVector::Constant(this->numHidden, 0.5);
	
	conInput << contextUnits, input;

	/* Forward propagation */
	//cout << "Forward propagation" << endl;
	hidOuts = (this->hiddenWeights.rowwise()*conInput.transpose()).rowwise().sum();
	hidOuts = sigmoid(hidOuts);
	outs = this->outputWeights.matrix()*hidOuts.matrix();
	outs = sigmoid(outs);

	/* Determine errors */
	outErrors = output - outs;

	/* Update context units */
	contextUnits = hidOuts;

	if(verbose){
		cout << "Inputs: " << endl;
		printMPMatrix(input.transpose());

		cout << "Outputs: " << endl;
		printMPMatrix(outs.transpose());

		cout << "Desired: " << endl;
		printMPMatrix(output.transpose());
	}

	/* Record mean squared error */
	error = outErrors.matrix().dot(outErrors.matrix())/outErrors.rows();
	return error;
}

MPVector &ElmanNet::MPRealToUnary(mpreal val, MPVector &arr){
	int i;
	for(i = 0; i < arr.rows(); i++){
		if(val.toLong() == i)
			arr(i) = 1;
		else
			arr(i) = 0;
	}
	return arr;
}

ElmanNet::~ElmanNet() {
	// TODO Auto-generated destructor stub
}
