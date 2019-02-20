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
