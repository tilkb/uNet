#include "FCNet.hpp"
#include "Dataset.hpp"

class SGDTrainer{
    int batch_size; 
    float lr;
    FCNet* network;
    Loss* loss;

public:
    SGDTrainer(int batch_size, float learning_rate, FCNet &model, Loss &loss){
        this->batch_size=batch_size;
        this->lr=learning_rate;
        this->network=&model;
        this->loss=&loss;
    }

    void step(){
        for (int i=0;i<this->network->bias.size();i++){
            this->network->bias[i]=this->network->bias[i] + (this->network->gradient_bias[i]*(-this->lr));
            this->network->matrix[i]=this->network->matrix[i] + this->network->gradient_matrix[i]*(-this->lr);
        }
    }

    void train(Dataset &dataset, int epochs){
        for (int epoch=0;epoch<epochs;epoch++){
            std::cout<<"Epoch "<<epoch+1<<std::endl;
            train_epoch(dataset);
        }
    }


    void train_epoch(Dataset &dataset){
        int id=0;
        float l=0.0;
        while (id<dataset.size()){
            int subset_id=0;
            this->network->zero();
            
            while (subset_id<this->batch_size && id<dataset.size()){
                auto [data, label] = dataset[id];
                this->network->forward(data);
                l += this->network->backward((*loss),label);
                id++;
                subset_id++;
            }
            this->step();
        }
        std::cout<<"Average loss:"<<l/dataset.size()<<std::endl;
    }
};