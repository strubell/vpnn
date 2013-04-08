/*
 * NeuralNet.cpp
 *
 *  Created on: Mar 23, 2013
 *      Author: ema
 */

#include "NeuralNet.h"

using namespace std;

NeuralNet::NeuralNet(int numInput, int numHidden, int numOutput, unsigned long initialPrecision, mpreal errorTol){
	this->numInput = numInput;
	this->numHidden = numHidden;
	this->numOutput = numOutput;
	this->errorTol = errorTol;
	setPrecision(initialPrecision);
}

/* Set the bitwise precision to be used by the network */
void NeuralNet::setPrecision(unsigned long prec){
	this->currentPrecision = prec;
	mpreal::set_default_prec(prec);
	cout.precision(bits2digits(prec));
}

MPVector NeuralNet::sigmoid(MPVector vec){
	return 1.0/(1.0+exp(-vec));
}

void NeuralNet::printMPMatrix(MPMatrix m){
	int i, j;
	for(i = 0; i < m.rows(); ++i){
		for(j = 0; j < m.cols(); ++j){
			cout << m(i,j) << " ";
		}
		cout << endl;
	}
}

NeuralNet::~NeuralNet(){

}
