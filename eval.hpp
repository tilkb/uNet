class Evaluator{
FCNet* network;
public:
    Evaluator(FCNet &network){
        this->network=&network;
    }
    float accuracy(Dataset &ds){
        float sum=0.0;
        for (int i=0;i<ds.size();i++){
            auto [data, label] = ds[i];
            Matrix predicted = this->network->forward(data);
            auto [predx, predy] = predicted.argmax();
            auto [targetx, targety] = label.argmax();
            if (targety==predy){
                sum=sum+1;
            }
        }
        return sum/((float)ds.size());
    }
};