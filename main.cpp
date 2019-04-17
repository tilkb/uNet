#include "SGDTrainer.hpp"
#include "eval.hpp"
#include <iostream>


FCNet model1(Dataset &dataset){
    std::vector<int> neurons1{ 400,50,10};
    std::vector<std::unique_ptr<uFunction>> activations1;
    activations1.push_back(std::unique_ptr<ReLu>(new ReLu()));
    activations1.push_back(std::unique_ptr<ReLu>(new ReLu()));
    activations1.push_back(std::unique_ptr<Softmax>(new Softmax()));
    FCNet model1(dataset.input_dim(),neurons1,activations1);
    CrossEntropyLoss loss;
    SGDTrainer sgd(8,0.01,model1,loss);
    sgd.train(dataset,10);
    return model1;
}

FCNet model2(Dataset &dataset){
    std::vector<int> neurons2{ 800, 10};
    std::vector<std::unique_ptr<uFunction>> activations2;
    activations2.push_back(std::unique_ptr<Sigmoid>(new Sigmoid()));
    activations2.push_back(std::unique_ptr<Softmax>(new Softmax()));
    FCNet model2(dataset.input_dim(),neurons2,activations2);
    CrossEntropyLoss loss;
    SGDTrainer sgd(8,0.01,model2,loss);
    sgd.train(dataset,0);
    return model2;
}

void eval(FCNet &network,Dataset &dataset, std::string name){
    Evaluator eval(network);
    float acc = eval.accuracy(dataset)*100.0;
    std::cout<<"Accuracy of "<<name<<": "<<acc<<"%"<<std::endl;
}

int main(){
    MNIST train("data/train-images-idx3-ubyte","data/train-labels-idx1-ubyte");
    std::cout<<"Train data loaded"<<std::endl;
    MNIST test("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    std::cout<<"Test data loaded"<<std::endl;
    std::cout<<"----------------------Model1--------------------"<<std::endl;
    FCNet m1 = model1(train);
    eval(m1,test,"Model1, deep ReLu activaion");
    std::cout<<"----------------------Model2--------------------"<<std::endl;
    FCNet m2 = model2(train);
    eval(m2,test,"Model2, wide sigmoid activation");
}