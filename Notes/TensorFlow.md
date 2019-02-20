# Tensorflow Neural Networks

-Python, Go, Java, C

-Training neural networks can create bottlebecks since it usually involves matrix multiplication --> GPU is faster than CPU --> Tensorflow supports both CPU and GPU programming --> also has specialized cloud computing

-Tensors are the basic data structures in Tensorflow which store data in any number of dimensions 
    1. Constants: immutable type of tensor --> can be used as nodes without inputs, outputting single value they store
    2. Variables: mutable type of tensor whose value can change during the run of a graph --> usually store the parameters            which need to be optimized (weights) --> Variables need to be initialized before running the graph
    3. Placeholders: tensors that store data from an external source --> value filled with graph is run --> used for inputting        data into the learning model
    
-Computational graph-a series of operatopns arranged into a graph of nodes --> each node may have multiple tensors as inputs and performs operations on tehm in order to calculate an output --> output could be input for another node --> neural network structure

-graph runs inside a session --> placeholders revieve concrete values through the feedback attributr

stackabuse.com/tensorflow-neural-network-tutorial

# Python Tensorflow Tutorial

-Tensors can inder data types --> need to be set for placeholders

-Tensorflow allows calculations to be split up among CPU and GPU

-Tensors are only created when ran --> to run operations between variables need to start a Tensorflow Session

-Tf.Session supplied a dictionary when run to fill placeholders

-Need weights and biases for each layer

## Neural Network Example

-Set up placeholders for training data

-Set up weights and biases for the layers (L-1 for L layers)

-Set up output of hidden layer

        z = xW + b
            (input vector)*(weights) + biases
            
        h = f(z)  --> activation function
-Do for every layer except input

-Include cost/loss function for optimization

-Gradient Descent optimizer provided by Tensorflow

adventuresinmacinelearning.com/python-tensorflow-tutorial

# Tensorflow: a universal approximator inside a neural network

-Universal Approximation Theorem
    -Any continuous function f is defined in R^n
    -Can find a wide enough 1-hidden layer neural network
    -that will approximate f to any accuracy on a closed interval
    
-Any continsuous function on a compact set (closed interval) can be approximated by a piecewise constant function to any accuracy

-Can build a neural network manually which will be as close as wanted to the piecewise function by adding as many nuerons as necessary

-As the number of neurons increses the approximation gets better

blog.metaflow.fr/tensorflow-howto-a-universal-approximator-inside-a-neural-net-bb034430b71e
