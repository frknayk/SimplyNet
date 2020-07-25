# SimplyNet

## About <a name = "about"></a>

- Numpy based flexible open-source neural-network library. 
- SimplyNet's backpropagation module is driven by coursera deep learning specialization course.


## Prerequisites

Numpy - 1.18.4


## Usage <a name = "usage"></a>

- Create neural network architecture on yaml file, then give its path to module.
- SimplyNet will automatically understand the architecture and create graph.

```
fcnn = FullyConnectedNetwork(input_size,output_size,seed=1,arch_path="path_to_architecture_yaml")
```

```
Run and check the example : python3 Examples/BinaryClassification_Example.py
```

```
Good-2-Know : According to the convention of SimplyNet, always use (input_size x num_of_samples) shape
```

## TODO
- Add cupy or any cuda supported GPU accelaration library for numpy 
- Add saving weights option to yaml config
- Make connection between tensorboard
- Re-implement logistic regression with SimplyNet modules

## Contributing
- Please send me an email for any idea or feedback about the project to furkanayik@outlook.com
