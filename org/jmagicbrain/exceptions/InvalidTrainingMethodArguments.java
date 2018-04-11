package org.jmagicbrain.exceptions;

public class InvalidTrainingMethodArguments extends RuntimeException {
    private static final String NAME = "InvalidTrainingMethodArguments";

    public InvalidTrainingMethodArguments(String message){
        super(message);
    }

    @Override
    public String toString() {
        return String.format("%s: %s", NAME, super.toString());
    }
}
