#include <tuple>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

class Dataset{
public:
    virtual int size()=0;
    virtual int input_dim()=0;
    virtual std::tuple<Matrix, Matrix> operator[](int id)=0;
};

class MNIST: public Dataset{
    std::vector<Matrix> data;
    std::vector<Matrix> label;
    
    void normalize(){
        float mean=0.0;
        float std=0.0;
        for(Matrix img : data){
            mean+=img.sum()/(img.row*img.column);
        }
        mean=mean/(this->data.size());

        for(Matrix img : data){
            for(int r=0;r<img.row;r++)
            {
                for(int c=0;c<img.column;c++){
                    std+=pow(img(r,c)-mean,2);
                }
            }
        }
        std=std/(this->data.size()*this->data[0].row*this->data[0].column -1);
        std=sqrt(std);

        for(int i=0;i<this->data.size();i++){
            Matrix img=this->data[i];
            this->data[i]=(img+(-mean))*(1/std);
        }
    }

    int ReverseInt (int i)
    {
        unsigned char ch1, ch2, ch3, ch4;
        ch1=i&255;
        ch2=(i>>8)&255;
        ch3=(i>>16)&255;
        ch4=(i>>24)&255;
        return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
    }
    void read_label(std::string path){
        std::ifstream file;
        file.open(path, std::ios::in | std::ios::binary);
        if (file.is_open()){ 
            int magic_number=0;
            int nr_images=0;

            file.read((char*)&magic_number,sizeof(magic_number));
            magic_number = this->ReverseInt(magic_number);

            file.read((char*)&nr_images,sizeof(nr_images));
            nr_images = this->ReverseInt(nr_images);

            for(int i=0;i<nr_images;i++)
            {
                Matrix one_hot_label(1,10);
                one_hot_label.zero();
                unsigned char tmp=0;
                file.read((char*)&tmp,sizeof(tmp));
                one_hot_label(0,tmp)=1.0;
                this->label.push_back(one_hot_label);
            }
        }
    }

    void read_data(std::string path){
        std::ifstream file;
        file.open(path, std::ios::in | std::ios::binary);
        if (file.is_open()){
            int magic_number=0;
            int nr_images=0;
            int rows=0;
            int cols=0;

            file.read((char*)&magic_number,sizeof(magic_number));
            magic_number = this->ReverseInt(magic_number);

            file.read((char*)&nr_images,sizeof(nr_images));
            nr_images = this->ReverseInt(nr_images);

            file.read((char*)&rows,sizeof(rows));
            rows = this->ReverseInt(rows);

            file.read((char*)&cols,sizeof(cols));
            cols = this->ReverseInt(cols);
            
            for(int i=0;i<nr_images;i++)
            {
                Matrix mat(1,rows*cols);
                for(int r=0;r<rows;r++)
                {
                    for(int c=0;c<cols;c++){
                        unsigned char tmp=0;
                        file.read((char*)&tmp,sizeof(tmp));
                        mat(0,r*rows+c)=(float)tmp;
                    }
                }
                data.push_back(mat);
            }
        }
        file.close();
    }

 public:
    MNIST(std::string data, std::string label){
        this->read_data(data);
        this->read_label(label);
        this->normalize();
    }

    virtual int size(){
        return this->data.size();
    }

    virtual int input_dim(){
        if (this->data.size()==0)
            return 0;
        else
            return this->data[0].row*this->data[0].column;
    }

    virtual std::tuple<Matrix, Matrix> operator[](int id){
        assert(id<this->data.size());
        return std::make_tuple(this->data[id],this->label[id]);
    }
};