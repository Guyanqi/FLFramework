#!/bin/sh
# STRING="Downloading the MNIST-data set and creating clients"
# echo $STRING
#eval mkdir MNIST_original
#eval cd MNIST_original
#eval curl -O "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
#eval curl -O "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
#eval curl -O "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
#eval curl -O "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
#eval gunzip train-images-idx3-ubyte.gz
#eval gunzip train-labels-idx1-ubyte.gz
#eval gunzip t10k-images-idx3-ubyte.gz
#eval gunzip t10k-labels-idx1-ubyte.gz
#eval cd ..
eval python Create_clients.py 
#STRING2="You can now run differentially private federated learning on the MNIST data set. Type python sample.py —-h for help"
#echo $STRING2
#STRING3="An example: …python sample.py -N 100… would run differentially private federated learning on 100 clients for a privacy budget of (epsilon, delta)"
#echo $STRING3
python sample.py -N 100
#python sample.py -N 1000
#python sample.py -N 10000
#STINRG4="For more information on how to use the the functions please refer to their documentation"
#echo $STRING4
