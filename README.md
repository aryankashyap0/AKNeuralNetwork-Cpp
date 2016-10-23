# AKNeuralNetwork-Cpp
**AKNeuralNetwork-Cpp** is a free open source lightweight and user friendly C++ library which implements multilayer artificial neural networks.

## AKNeuralNetwork-Cpp Features
* Multilayer Artificial Neural Network Library in C++
* Supports Forward and Back Propogation.
* Evolving topology training
* Easy to use (Example Below)
* Fast
* Easy to load Large Datasets (in the machine for now)
* Default Data sets included for you to try!

## To Add
To Clone: `git clone https://github.com/NSAryan12/AKNeuralNetwork-Cpp.git'

## Example How to use


'
#include <stdio.h>
#include "AKNeuralNetwork.cpp"

int main(){
    
    TrainingData trainData("Here goes the PATH to the training data .txt file..... Example: /Users/Aryan/Desktop/out_xor.txt");
    
    // e.g., { 3, 2, 1 }
    std::vector<unsigned> topology;
    trainData.getTopology(topology);
    
    Net myNet(topology);
    
    t_vals inputVals, targetVals, resultVals;
    int trainingPass = 0;
    
    while (!trainData.isEof())
    {
        ++trainingPass;
        std::cout << std::endl << "Pass " << trainingPass << std::endl;
        
        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0])
            break;
        
        showVectorVals("Inputs:", inputVals);
        myNet.feedForward(inputVals);
        
        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);
        
        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());
        
        myNet.backProp(targetVals);
        
        // Report how well the training is working, average over recent samples:
        std::cout << "Net current error: " << myNet.getError() << std::endl;
        std::cout << "Net recent average error: " << myNet.getRecentAverageError() << std::endl;
        
        if (trainingPass > 100 && myNet.getRecentAverageError() < 0.05)
        {
            std::cout << std::endl << "average error acceptable -> break" << std::endl;
            break;
        }
    }
    
    std::cout << std::endl << "Done" << std::endl;
    
    if (topology[0] == 2)
    {
        std::cout << "TEST" << std::endl;
        std::cout << std::endl;
        
        unsigned dblarr_test[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
        
        for (unsigned i = 0; i < 4; ++i)
        {
            inputVals.clear();
            inputVals.push_back(dblarr_test[i][0]);
            inputVals.push_back(dblarr_test[i][1]);
            
            myNet.feedForward(inputVals);
            myNet.getResults(resultVals);
            
            showVectorVals("Inputs:", inputVals);
            showVectorVals("Outputs:", resultVals);
            
            std::cout << std::endl;
        }
        
        std::cout << "/TEST" << std::endl;
    }
    
}'
