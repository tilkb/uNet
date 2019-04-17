#include <vector>
#include <memory>
#include "core.hpp"

class FCNet{
public:
        std::vector<Matrix> bias;
        std::vector<Matrix> matrix;
        std::vector<Matrix> gradient_matrix;
        std::vector<Matrix> gradient_bias;
        std::vector<Matrix> stored_output;
        std::vector<Matrix> stored_before_activation;
        std::vector<std::unique_ptr<uFunction>> activation;

        FCNet(int input_dim, std::vector<int> neuron_numbers, std::vector<std::unique_ptr<uFunction>> &activation){
            assert(neuron_numbers.size() == activation.size());
            int last_layer_neuron=input_dim;
            for (int i=0;i<activation.size();i++)
                this->activation.push_back(std::move(activation[i]));
            this->stored_output.push_back(Matrix(1,input_dim));
            for (int neuron : neuron_numbers){
                this->stored_output.push_back(Matrix(1,neuron));
                this->stored_before_activation.push_back(Matrix(1,neuron));
                this->bias.push_back(Matrix(1,neuron));
                this->bias.back().init_normal_random();
                this->gradient_bias.push_back(Matrix(1,neuron));
                this->matrix.push_back(Matrix(last_layer_neuron, neuron));
                this->matrix.back().init_normal_random();
                this->gradient_matrix.push_back(Matrix(last_layer_neuron, neuron));
                last_layer_neuron=neuron;
            }
            
        }

        void zero(){
            for (int i=0;i<this->gradient_bias.size();i++){
                this->gradient_bias[i].zero();
                this->gradient_matrix[i].zero();
            }
        }
        
        Matrix forward(Matrix& input){
            Matrix current_input=input;
            for (int i=0;i<this->bias.size();i++){
                this->stored_output[i]=current_input;
                current_input = (current_input*this->matrix[i]) + current_input +this->bias[i];
                this->stored_before_activation[i]=current_input;
                current_input = this->activation[i]->forward(current_input);
            }
            this->stored_output[this->bias.size()]=current_input;
            return current_input;
        }
        float backward(Loss &loss, Matrix &target){
            float L= loss.forward(this->stored_output[this->stored_output.size()-1],target);
            //Handle softmax+cross-entropy numerical problems
            Matrix last_grad(1,1);
            if (typeid(loss)==typeid(CrossEntropyLoss) ){
                CrossEntropyLoss* ce_loss = dynamic_cast<CrossEntropyLoss*>(&loss);
                last_grad=ce_loss->backward_logit(this->stored_output[this->stored_output.size()-1],target);
            }
            else{
                last_grad=loss.backward(this->stored_output[this->stored_output.size()-1],target);
            }
            for (int i=this->bias.size()-1;i>=0;i--){
                if (i<this->bias.size()-1 || typeid(loss)!=typeid(CrossEntropyLoss))
                    last_grad = last_grad.hadamard_product(this->activation[i]->backward(this->stored_before_activation[i]));
                this->gradient_matrix[i]= this->gradient_matrix[i] + (this->stored_output[i].transpose() * last_grad);
                this->gradient_bias[i]= this->gradient_bias[i] + last_grad;
                last_grad = (this->matrix[i]*last_grad.transpose()).transpose();
            }
            return L;
        }
        


};