#include <math.h>
#include "Matrix.hpp"

class uFunction{
public:
    virtual Matrix forward(Matrix &input)=0;
    virtual Matrix backward(Matrix &input)=0;
};

class Loss{
public:
    virtual float forward(Matrix &predicted, Matrix &target)=0;
    virtual Matrix backward(Matrix &predicted, Matrix &target)=0;
};

class ReLu: public uFunction{
public:
    virtual Matrix forward(Matrix &input){
        Matrix result=input;
        for (int i=0;i<input.row;i++){
            for (int j=0;j<input.column;j++){
                if (result(i,j)<0){
                    result(i,j)=0.0;
                }
            }
        }
        return result;
    }

     virtual Matrix backward(Matrix &input){
        Matrix result=Matrix(input.row, input.column);
        for (int i=0;i<input.row;i++){
            for (int j=0;j<input.column;j++){
                if (input(i,j)<0){
                    result(i,j)=0.0;
                }
                else{
                    result(i,j)=1.0;
                }
            }
        }
        return result;
     }
};

class Sigmoid: public uFunction{
public:
    virtual Matrix forward(Matrix &input){
        Matrix result(input.row,input.column);
        for (int i=0;i<input.row;i++){
            for (int j=0;j<input.column;j++)
                result(i,j)=1.0/(1.0 + exp(-input(i,j)));
        }
        return result;
    }

    virtual Matrix backward(Matrix &input){
        Matrix sigm = this->forward(input);
        for (int i=0;i<input.row;i++){
            for (int j=0;j<input.column;j++){
                sigm(i,j) = sigm(i,j)*(1-sigm(i,j));
            }
        }
        return sigm;
    }
};

class Softmax: public uFunction{
public:
    virtual Matrix forward(Matrix &input){

        Matrix result(input.row,input.column);
        float normalization=0.0;
        auto [max_i, max_j] = input.argmax();
        for (int i=0;i<input.row;i++){
            for (int j=0;j<input.column;j++){
                normalization+=exp(input(i,j)-input(max_i,max_j));
            }
        }

        for (int i=0;i<input.row;i++){
            for (int j=0;j<input.column;j++)
                result(i,j)=exp(input(i,j)-input(max_i,max_j))/normalization;
        }
        return result;   
    }

    virtual Matrix backward(Matrix &input){
        Matrix result = this->forward(input);
        for (int i=0;i<input.row;i++){
            for (int j=0;j<input.column;j++){
                result(i,j) = result(i,j)*(1-result(i,j));
            }
        }
        return result;
    }
};

class CrossEntropyLoss: public Loss{
public:
    virtual float forward(Matrix &predicted, Matrix &target){
        assert(predicted.row==target.row && predicted.column==target.column);
        float sum_loss=0.0;
        for (int i=0;i<predicted.row;i++){
            for (int j=0;j<predicted.column;j++){
                // eliminate log0 problem
                float pred=predicted(i,j);
                if (pred<0.000001)
                    pred=0.000001;
                sum_loss+= -target(i,j)*log(pred);
            }
        }
        return sum_loss;
    }

    virtual Matrix backward(Matrix &predicted, Matrix &target){
        assert(predicted.row==target.row && predicted.column==target.column);
        Matrix result = target; 
        for (int i=0;i<predicted.row;i++){
            for (int j=0;j<predicted.column;j++){
                result(i,j)=-result(i,j)/target(i,j);
            }
        }
        return result;
    }

    Matrix backward_logit(Matrix &predicted, Matrix &target){
        assert(predicted.row==target.row && predicted.column==target.column);
        return predicted + (target*(-1));
    }

};