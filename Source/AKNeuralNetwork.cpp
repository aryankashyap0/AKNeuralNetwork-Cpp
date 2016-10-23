//
//  main.cpp
//  AKNeuralNetwork-C++
//
//  Created by Aryan Kashyap on 21/10/2016.
//  Copyright © 2016 Aryan Kashyap. All rights reserved.
//

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

// using namespace std;

typedef std::vector<unsigned long> t_vals;

#pragma mark - Training Data

// Silly class to read training data from a text file - Replace This.
// Replace class TrainingData with whatever you need to get input data into the
// program, e.g., connect to a database, or take a stream of data from stdin, or
// from a file specified by a command line argument, etc.
// We will be adding this in the future

class TrainingData
{
public:
    TrainingData(const std::string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(std::vector<unsigned> &topology);
    
    // Returns the number of input values read from the file:
    unsigned getNextInputs(t_vals &inputVals);
    unsigned getTargetOutputs(t_vals &targetOutputVals);
    
private:
    std::ifstream m_trainingDataFile;
};

#pragma mark - Get topology value
void TrainingData::getTopology(std::vector<unsigned> &topology)
{
    std::string line;
    std::string label;
    
    getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }
    
    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
    
    return;
}

TrainingData::TrainingData(const std::string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

#pragma mark - Get next inputs
unsigned TrainingData::getNextInputs(t_vals &inputVals)
{
    inputVals.clear();
    
    std::string line;
    getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    
    std::string label;
    ss>> label;
    if (label.compare("in:") == 0)
    {
        double oneValue;
        
        while (ss >> oneValue)
            inputVals.push_back(oneValue);
    }
    
    return inputVals.size();
}

#pragma mark - Get the target outuputs
unsigned TrainingData::getTargetOutputs(t_vals &targetOutputVals)
{
    targetOutputVals.clear();
    
    std::string line;
    getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    
    std::string label;
    ss >> label;
    if (label.compare("out:") == 0)
    {
        double oneValue;
        
        while (ss >> oneValue)
            targetOutputVals.push_back(oneValue);
    }
    
    return targetOutputVals.size();
}


typedef std::vector<unsigned long> t_vals;


#pragma mark - Connection Struct

// Connection
struct Connection
{
    double weight;
    double deltaWeight;
};

typedef std::vector<Connection> Connections;

#pragma mark - Neuron

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    
    inline void setOutputVal(double val) { m_outputVal = val; }
    inline double getOutputVal(void) const { return m_outputVal; }
    
    void feedForward(const Layer& prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer& nextLayer);
    void updateInputWeights(Layer& prevLayer);
    
private:
    // [0.0..1.0] overall net training rate
    static double eta;
    // [0.0..n] multiplier of last weight change (momentum)
    static double alpha;
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    Connections m_outputWeights;
    unsigned m_myIndex;
    double m_gradient; // used by the backpropagation
};

// overall net learning rate [0.0..1.0]
double Neuron::eta = 0.15;

// momentum, multiplier of last deltaWeight, [0.0..1.0]

double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
: m_myIndex(myIndex)
{
    for (unsigned i = 0; i < numOutputs; ++i)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
}

#pragma mark - Feed Forward the algorithm
void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    
    // Sum the previous Layer's output (which are our inputs)
    // Include the bias node from the previous layer
    
    for(unsigned n = 0; n < prevLayer.size(); n++){
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
        m_outputVal = Neuron::transferFunction(sum);
    }
}

#pragma mark - Calculate Output Gradients
void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}


#pragma mark - Calculate Hidden Gradients
void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

#pragma mark - Update the input weight
void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    
    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        
        double newDeltaWeight =
        // Individual input, magnified by the gradient and train rate:
        eta
        * neuron.getOutputVal()
        * m_gradient
        // Also add momentum = a fraction of the previous delta weight;
        + alpha
        * oldDeltaWeight
        ;
        
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

#pragma mark - Transfer function
double Neuron::transferFunction(double x)
{
    // tanh - output range [-1.0..1.0]
    return tanh(x);
}

#pragma mark - Transfer function derivative
double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    return (1.0 - x * x);
}

#pragma mark - SumDOW
double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    
    // Sum our contributions of the errors at the nodes we feed.
    
    // exclude bias neuron
    unsigned num_neuron = (nextLayer.size() - 1);
    
    for (unsigned n = 0; n < num_neuron; ++n)
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    
    return sum;
}

#pragma mark - Net

class Net
{
public:
    Net(const std::vector<unsigned> &topology);
    void feedForward(const t_vals &inputVals);
    void backProp(const t_vals &targetVals);
    void getResults(t_vals &resultVals) const;
    
public: // error
    double getError(void) const { return m_error; }
    double getRecentAverageError(void) const { return m_recentAvgError; }
    
private:
    std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    
    // error
    double m_error;
    double m_recentAvgError;
    static double k_recentAvgSmoothingFactor;
    // /error
};

double Net::k_recentAvgSmoothingFactor = 100.0; // Number of training sample to average over

Net::Net(const std::vector<unsigned>& topology)
: m_error(0.0),
m_recentAvgError(0.0)
{
    assert(!topology.empty()); // no empty topology
    
    for (unsigned i  = 0; i < topology.size(); i++){
        unsigned num_neuron = topology[i];
        
        assert(num_neuron > 0); // no empty layer
        
        m_layers.push_back(Layer());
        
        Layer& new_layer = m_layers.back();
        
        bool is_last_layer = (i == (topology.size() - 1));
        
        // 0 output if on the last layer
        unsigned numOutputs = ((is_last_layer) ? (0) : (topology[i + 1]));
        
        // We have a new layer, now fill it with neurons, and
        // add a bias neuron in each layer.
        for (unsigned j = 0; j < (num_neuron + 1); ++j) // add a bias neuron
            new_layer.push_back( Neuron(numOutputs, j) );
        
        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        Neuron& bias_neuron = new_layer.back();
        bias_neuron.setOutputVal(1.0);
    }
}

#pragma mark - Feed forward in the neural network
void Net::feedForward(const t_vals &inputVals)
{
    assert(inputVals.size() == (m_layers[0].size() - 1)); //exclude the bias neuron
    
    // Assign (latch) the input values into input Neurons
    
    // forward propgate
    for(unsigned i = 1; i < m_layers.size(); i++){
        Layer& prevLayer = m_layers[i - 1];
        Layer& currLayer = m_layers[i];
        
        
        unsigned num_neuron = (currLayer.size() - 1); // exclude bias neuron
        for (unsigned n = 0; n < num_neuron; n++) {
            currLayer[n].feedForward(prevLayer);
        }
    }
}

#pragma mark - Back propogate in the Neural Network
void Net::backProp(const t_vals &targetVals)
{
    
    // Calculate overall net error (RMS of output neuron errors)
    
    Layer& outputLayer = m_layers.back();
    m_error = 0.0;
    
    for(unsigned n = 0; n < outputLayer.size() - 1; n++)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    
    m_error /= (outputLayer.size() - 1); // get average error squared
    m_error = sqrt(m_error); // RMS
    
    // Implement a recent average meausurment
    m_recentAvgError = (m_recentAvgError * k_recentAvgSmoothingFactor + m_error) / (k_recentAvgSmoothingFactor + 1.0);
    
    // Gradients
    // Calculate output layer gradients
    for (unsigned i = (m_layers.size() - 2); i > 0; --i)
    {
        Layer &hiddenLayer = m_layers[i];
        Layer &nextLayer = m_layers[i + 1];
        
        for (unsigned n = 0; n < hiddenLayer.size(); ++n)
            hiddenLayer[n].calcHiddenGradients(nextLayer);
    }
    
    // For all layers from outputs to first hidden layer,
    // update connection weights
    
    for (unsigned i = (m_layers.size() - 1); i > 0; --i)
    {
        Layer &currLayer = m_layers[i];
        Layer &prevLayer = m_layers[i - 1];
        
        for (unsigned n = 0; n < (currLayer.size() - 1); ++n) // exclude bias
            currLayer[n].updateInputWeights(prevLayer);
    }
}

#pragma mark - Get the results :)
void Net::getResults(t_vals &resultVals) const
{
    resultVals.clear();
    
    const Layer& outputLayer = m_layers.back();
    
    // exclude last neuron (bias neuron)
    unsigned total_neuron = (outputLayer.size() - 1);
    
    for (unsigned n = 0; n < total_neuron; ++n)
        resultVals.push_back(outputLayer[n].getOutputVal());
}

#pragma mark - Main Functions

void showVectorVals(std::string label, t_vals &v)
{
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i)
        std::cout << v[i] << " ";
    
    std::cout << std::endl;
}

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
    
}
