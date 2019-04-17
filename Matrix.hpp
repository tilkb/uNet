#include <stdexcept>
#include <algorithm>
#include <assert.h>
#include <random>
#include <math.h>
#include <iostream>
#include <tuple>

class Matrix{
        float* data;
public:
        int row;
        int column;
         
        Matrix(const int row,const  int column){
            this->row = row;
            this->column = column;
            this->data = new float[row * column];
        };

        float& operator()(int i, int j){
            if ((this->row>row) && (this->column>column)){
                throw std::invalid_argument("Overindexing!");
            }
            return this->data[i*this->column + j];
        }

        Matrix(const Matrix &m){
            this->row = m.row;
            this->column = m.column;
            
            this->data = new float[m.row* m.column];  
            std::copy(m.data, m.data + m.row*m.column,this->data);
        };

        Matrix& operator=(const Matrix &m){
            if (this->row!=m.row || this->column!=m.column){
                delete[] this->data;
                this->data = new float[m.row* m.column];
            }
            this->row=m.row;
            this->column=m.column;
            std::copy(m.data, m.data + m.row*m.column,this->data);
            return *this;
        }

        void init_normal_random(){
            std::default_random_engine generator;
            //use Xavier initialization
            std::normal_distribution<float> distribution(0.0,sqrt(1.0/(this->row+this->column)));
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    (*this)(i,j) = distribution(generator);
                }
            }
        }

        void zero(){
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    (*this)(i,j) = 0.0;
                }
            }
        };

        Matrix operator+(Matrix m2){
            if ((this->row!=m2.row) && (this->column!=m2.column)){
                throw std::invalid_argument("Size doesn't match(add)");
            }
            Matrix result(this->row,this->column);
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    result(i,j) = (*this)(i,j)+m2(i,j);
                }
            }
            return result;
        };

        Matrix operator*(Matrix m2){
            if (this->column!=m2.row){
                throw std::invalid_argument("Size doesn't match(multiplication)");
            }

            Matrix result(this->row,m2.column);
            for (int i=0;i<this->row;i++){
                for (int j=0;j<m2.column;j++){
                    result(i,j)=0.0;
                    for (int k=0;k<this->column;k++){
                        result(i,j) += (*this)(i,k) * m2(k,j);
                    }
                }
            }
            return result;
        };
        Matrix operator*(float c){
            Matrix result = *this;
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    result(i,j)*=c;
                }
            }
            return result;
        }

        Matrix operator+(float c){
            Matrix result = *this;
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    result(i,j)+=c;
                }
            }
            return result;
        }

        Matrix hadamard_product(Matrix m){
            assert(m.row==(*this).row && m.column==(*this).column);
            Matrix result = *this;
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    result(i,j)=result(i,j) * m(i,j);
                }
            }
            return result;
        }

        Matrix transpose(){
            Matrix result(this->column,this->row);
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    result(j,i)=(*this)(i,j);
                }
            }
            return result;
        }

        float sum(){
            float result=0.0;
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    result+=(*this)(i,j);
                }
            }
            return result;
        }

        std::tuple<int,int> argmax(){
            int argx=0;
            int argy=0;
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    if ((*this)(argx,argy)<(*this)(i,j)){
                        argx=i;
                        argy=j;
                    }
                }
            }
            return std::make_tuple(argx,argy);
        }


        ~Matrix(){
            delete[] this->data;
        }
};