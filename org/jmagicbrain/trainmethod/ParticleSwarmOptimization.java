package org.jmagicbrain.trainmethod;

import java.util.Random;

public class ParticleSwarmOptimization extends TrainMethod{

    private final int numberOfParticles;
    private final double maxX;
    private final double minX;

    private double bestGlobalError;
    private double[] bestGlobalPosition;

    private final Random random;
    private Particle[] swarm;

    public ParticleSwarmOptimization(int numberOfParticles, double maxX, double minX, double[][] trainingSet, double[][] expectedOutput) {
        super(trainingSet, expectedOutput);
        this.numberOfParticles = numberOfParticles;
        this.maxX = maxX;
        this.minX = minX;
        this.swarm = new Particle[this.numberOfParticles];
        bestGlobalError = Double.MAX_VALUE;

        this.random = new Random();
    }

    @Override
    public void train() {
        
    }

    private void initParticles(){
        double hi = 0.1 * maxX;
        double lo = 0.1* minX;
        double[] position = new double[neuralNetwork.getNumberOfWeights()];
        double[] velocity = new double[neuralNetwork.getNumberOfWeights()];
        double error;
        for(int i = 0; i < numberOfParticles; i++){
            for(int j = 0; j < position.length; j++) {
                position[j] = (maxX - minX) * random.nextDouble() - minX;
                error = this.errorFunction.getError(trainingSet, expectedOutput);
                velocity[j] = (hi- lo) * random.nextDouble() - lo;
            }
            swarm[i] = new Particle(position, error, velocity, position, error);
            if(error < bestGlobalError){
                bestGlobalError = error;
                System.arraycopy(position, 0, bestGlobalPosition, 0, bestGlobalPosition.length);
            }
        }
    }

    private class Particle{
        public double[] position;
        public double error;
        public double[] velocity;
        public double[] bestPosition;
        public double bestError;

        public Particle(double[] position, double error, double[] velocity, double[] bestPosition, double bestError) {
            System.arraycopy(this.position, 0, position, 0, position.length);
            this.error = error;
            System.arraycopy(this.velocity, 0, velocity, 0, velocity.length);
            System.arraycopy(this.bestPosition, 0, bestPosition, 0, bestPosition.length);
            this.bestError = bestError;
        }
    }
}
