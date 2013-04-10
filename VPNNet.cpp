
#include <iostream>
#include "FeedForwardNet.h"
#include "ElmanNet.h"
#include "AttentionNet.h"

#define INT_BITS 32

/* Convert an integer to an array of integers representing its bits */
int *intToBinary(int a, int *buff);

/* Test feed forward network (XOR) */
void testFeedForwardNet();

/* Test Elman network (couple of sequences) */
void testElmanNet();

void testAttentionNet();

int main(int argc, char **argv) {

	//testFeedForwardNet();
	//testElmanNet();
	testAttentionNet();

	return 0;
}

void testFeedForwardNet(){
	int i, j;

	/* Set up initial precision */
	unsigned long initialPrecision = 50;
	mpreal::set_default_prec(initialPrecision);
	std::cout.precision(bits2digits(initialPrecision));

	/* Training data hard coded for now */
	int inputSize, hiddenSize, outputSize;
	inputSize = 2;
	hiddenSize = 10;
	outputSize = 1;

	/* Read in training/testing data from file(s) */
	int trainingDataLen = static_cast<int>(std::pow(2.0,inputSize));
	int bitArr[INT_BITS];
	int numIterations = 1000;
	mpreal eta = 4.0;
	mpreal errorTol = 0.05;

	MPMatrix trainingIn(trainingDataLen, inputSize);
	MPMatrix trainingOut(trainingDataLen, outputSize);
	MPMatrix trainingErr(numIterations, outputSize);

	MPMatrix testingIn(trainingDataLen, inputSize);
	MPMatrix testingOut(trainingDataLen, outputSize);
	MPMatrix testingErr(numIterations, outputSize);

	for(i = 0; i < trainingDataLen; ++i){

		intToBinary(i, bitArr);

		for(j = inputSize; j > 0; --j){
			trainingIn(i,j-1) = bitArr[INT_BITS-j];
			testingIn(i,j-1) = bitArr[INT_BITS-j];
		}

		for(j = 0; j < outputSize; ++j){
			trainingOut(i,j) = (bitArr[INT_BITS-1] != bitArr[INT_BITS-2]);
			testingOut(i,j) = (bitArr[INT_BITS-1] != bitArr[INT_BITS-2]);
		}
	}

	printf("Training on input\n");
	for(i = 0; i < trainingDataLen; ++i){
		for(j = 0; j < inputSize; j++){
			std::cout << trainingIn(i,j);
		}
		std::cout << " -> ";
		for(j = 0; j < outputSize; ++j){
			std::cout << trainingOut(i,j);
		}
		std::cout << std::endl;
	}

	/* Train network on training data */
	FeedForwardNet ffNet(inputSize, hiddenSize, outputSize, initialPrecision, errorTol);
	ffNet.train(trainingIn, trainingOut, trainingErr, eta, numIterations);

	for(i = 0; i < numIterations; ++i){
		std::cout << trainingErr(i,0) << "\n";
	}

	/* Test network on testing data */
	ffNet.test(testingIn, testingOut, testingErr);
}

void testElmanNet(){
	int i, j;

	/* Set up initial precision */
	unsigned long initialPrecision = 50;
	mpreal::set_default_prec(initialPrecision);
	std::cout.precision(bits2digits(initialPrecision));

	/* Training data hard coded for now */
	int inputSize = 3;
	int hiddenSize = 4;
	int outputSize = 3;
	int sequenceLen = 3;
	int trainingDataLen = 2;
	int testingDataLen = trainingDataLen;
	int numIterations = 500;
	mpreal eta = 4.0;
	mpreal errorTol = 1e-2;

	MPMatrix trainingIn(trainingDataLen, sequenceLen);
	MPMatrix trainingOut(trainingDataLen, sequenceLen);
	MPMatrix trainingErr(numIterations, outputSize);

	MPMatrix testingIn(testingDataLen, sequenceLen);
	MPMatrix testingOut(testingDataLen, sequenceLen);
	MPMatrix testingErr(numIterations, outputSize);

	trainingIn << 0, 1, 2,
				  1, 0, 2;
	trainingOut << 1, 2, 0,
				   0, 2, 1;

	testingIn << 0, 1, 2,
				  1, 0, 2;
	testingOut << 1, 2, 0,
				   0, 2, 1;

	printf("Training on input: \n");
	for(i = 0; i < trainingDataLen; ++i){
		for(j = 0; j < sequenceLen; j++){
			std::cout << trainingIn(i,j);
		}
		std::cout << " -> ";
		for(j = 0; j < sequenceLen; ++j){
			std::cout << trainingOut(i,j);
		}
		std::cout << std::endl;
	}

	/* Train network on training data */
	ElmanNet elNet(inputSize, hiddenSize, outputSize, initialPrecision, errorTol);
	elNet.train(trainingIn, trainingOut, trainingErr, eta, numIterations);

	for(i = 0; i < numIterations; ++i){
		std::cout << trainingErr(i,0) << "\n";
	}

	/* Test network on testing data */
	elNet.test(testingIn, testingOut, testingErr);
}

void testAttentionNet(){
	int i, j;

	/* Set up initial precision */
	unsigned long initialPrecision = 32;
	mpreal::set_default_prec(initialPrecision);
	std::cout.precision(bits2digits(initialPrecision));

	/* Training data hard coded for now */
	int sequenceLen = 10;
	int inputSize = sequenceLen + 1;	// to encode distance to attend
	int hiddenSize = 20; 				// arbitrary guess
	int outputSize = 1;
	int trainingDataLen = static_cast<int>(std::pow(2.0,inputSize-3)); // also sort of arbitrary
	int testingDataLen = static_cast<int>(std::pow(2.0,inputSize)-std::pow(2.0,inputSize-3));  // the rest
	int numIterations = 500;			// also arbitrary
	mpreal eta = 4.0;					// ...
	mpreal errorTol = 1e-2;				// ...
	mpreal desiredAccuracy = 0.8;		// ...

	MPMatrix trainingIn(trainingDataLen, sequenceLen);
	MPMatrix trainingOut(trainingDataLen, sequenceLen);
	MPMatrix trainingErr(numIterations, outputSize);

	MPMatrix testingIn(testingDataLen, sequenceLen);
	MPMatrix testingOut(testingDataLen, sequenceLen);
	MPMatrix testingErr(numIterations, outputSize);

	trainingIn << 0, 1, 2,
				  1, 0, 2;
	trainingOut << 1, 2, 0,
				   0, 2, 1;

	testingIn << 0, 1, 2,
				  1, 0, 2;
	testingOut << 1, 2, 0,
				   0, 2, 1;

	printf("Training on input: \n");
	for(i = 0; i < trainingDataLen; ++i){
		for(j = 0; j < sequenceLen; j++){
			std::cout << trainingIn(i,j);
		}
		std::cout << " -> ";
		for(j = 0; j < sequenceLen; ++j){
			std::cout << trainingOut(i,j);
		}
		std::cout << std::endl;
	}

	/* Train network on training data */
	AttentionNet attentionNet(inputSize, hiddenSize, outputSize, initialPrecision, errorTol, desiredAccuracy);
	attentionNet.train(trainingIn, trainingOut, trainingErr, eta, numIterations);

	for(i = 0; i < numIterations; ++i){
		std::cout << trainingErr(i,0) << "\n";
	}

	/* Test network on testing data */
	attentionNet.test(testingIn, testingOut, testingErr);
}

/* Convert an integer to an array of integers representing its bits */
int *intToBinary(int a, int *buff){
	int i;
	for(i = INT_BITS-1; i >= 0; i--){
		buff[i] = (a & 1);
		a >>= 1;
	}
	return buff;
}


