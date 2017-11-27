package org.jmagicbrain.exceptions;

public class InvalidNeuralNetworkArguments extends RuntimeException {

    private static String NAME = "InvalidNeuralNetworkArguments";

    public InvalidNeuralNetworkArguments(String message){
        super(NAME + " " + message);
    }

    @Override
    public String toString() {
        return super.toString();
    }
}
