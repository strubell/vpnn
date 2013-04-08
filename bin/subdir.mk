################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../AttentionNet.cpp \
../ElmanNet.cpp \
../FeedForwardNet.cpp \
../NeuralNet.cpp \
../VPNNet.cpp 

OBJS += \
./AttentionNet.o \
./ElmanNet.o \
./FeedForwardNet.o \
./NeuralNet.o \
./VPNNet.o 

CPP_DEPS += \
./AttentionNet.d \
./ElmanNet.d \
./FeedForwardNet.d \
./NeuralNet.d \
./VPNNet.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


