# README

JMagicBrain is a Neural Network API

## Disclaimer

This is a school project, not for real world apps

## Tutorial

```Java
    ActivationFunction activationFunction = new Sigmoid();
    ErrorFunction errorFunction = new MeanSquaredError();
    WeightInitializer weightInitializer = new DefaultInitializer();
    
    TrainMethod trainMethod = new BackPropagation.BackPropagationBuilder()
        .setErrorFunction(errorFunction)
        .setMaxEpochs(1000)
        .setMaxError(0.09)
        .setMomentum(0.099099)
        .setLearningRate(0.031099)
        .setTrainingSet(images)
        .setExpectedOutput(expected)
        .build();
    
    neuralNetwork = new NeuralNetwork.NeuralNetworkBuilder()
        .setActivationFunction(activationFunction)
        .setErrorFunction(errorFunction)
        .setTrainingMethod(trainMethod)
        .setWeightInitializer(weightInitializer)
        .addLayer(784)
        .addLayer(16)
        .addLayer(10)
        .build();
        
    neuralNetwotk.train();
    

