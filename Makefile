compile:
	g++ main.cpp -std=c++17 -O3 -o main 
clean:
	rm main
download:
	mkdir data
	wget -P data http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	wget -P data http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	wget -P data http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	wget -P data http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	gunzip data/train-images-idx3-ubyte.gz
	gunzip data/train-labels-idx1-ubyte.gz
	gunzip data/t10k-images-idx3-ubyte.gz
	gunzip data/t10k-labels-idx1-ubyte.gz
	
