package org.jmagicbrain.exceptions;

public class InvalidNeuralNetworkArguments extends RuntimeException {

    private static final String NAME = "InvalidNeuralNetworkArguments";

    public InvalidNeuralNetworkArguments(String message){
        super(message);
    }

    @Override
    public String toString() {
        return String.format("%s: %s", NAME, super.toString());
    }
}
