package org.jmagicbrain.trainmethod;

import org.jmagicbrain.functions.ErrorFunction;

import java.util.Random;

public class ParticleSwarmOptimization extends TrainMethod{

    private final int numberOfParticles;
    private final double maxX;
    private final double minX;

    private double bestGlobalError;
    private double[] bestGlobalPosition;

    private final Random random;
    private Particle[] swarm;

    private final double W;
    private final double congitiveLocalConstant;
    public final double socialGlobalConstant;

    public ParticleSwarmOptimization(double congitiveLocalConstant, double socialGlobalConstant, int numberOfParticles, double maxX, double minX, int maxEpochs, double maxError, double[][] trainingSet, double[][] expectedOutput, ErrorFunction errorFunction) {
        super(maxEpochs, maxError, trainingSet, expectedOutput, errorFunction);
        this.numberOfParticles = numberOfParticles;
        this.maxX = maxX;
        this.minX = minX;
        this.W = 0.7;
        this.congitiveLocalConstant = congitiveLocalConstant;
        this.socialGlobalConstant = socialGlobalConstant;
        this.swarm = new Particle[this.numberOfParticles];
        bestGlobalError = Double.MAX_VALUE;

        this.random = new Random();
    }

    @Override
    public void train() {
        initParticles();
        int epoch = 0;
        double errt = 0;
        int[] sequence = new int[neuralNetwork.getNumberOfWeights()];
        for (int i = 0; i < sequence.length; i++){
            sequence[i] = i;
        }
        double newError;
        double r1, r2;
        boolean endEvaluation = false;
        Particle currentParticle;


        while (epoch < maxEpochs){
            if(bestGlobalError < maxError) break;
            for(int i = 0; i < sequence.length; i++){
                currentParticle = swarm[sequence[i]];
                for(int j = 0; j < neuralNetwork.getNumberOfWeights(); j++){
                    r1 = random.nextDouble();
                    r2 = random.nextDouble();

                    currentParticle.velocity[j] = (this.W * currentParticle.velocity[j]) +
                            ((r1 * this.congitiveLocalConstant) * (currentParticle.bestPosition[j] - currentParticle.position[j])) +
                            ((r2 * this.socialGlobalConstant) * (bestGlobalPosition[j] - currentParticle.position[j]));
                    currentParticle.position[j] = currentParticle.position[j] + currentParticle.velocity[i];
                    if(currentParticle.position[j] < minX) currentParticle.position[j] = minX;
                    if(currentParticle.position[j] > maxX) currentParticle.position[j] = maxX;
                }
                newError = errorFunction.getError(trainingSet, expectedOutput);
                if(newError < currentParticle.bestError){
                    currentParticle.bestError = newError;
                    System.arraycopy(currentParticle.position, 0 ,currentParticle.bestPosition, 0, currentParticle.position.length);
                }

                if(newError < bestGlobalError){
                    bestGlobalError = newError;
                    System.arraycopy(currentParticle.position, 0, bestGlobalPosition, 0, bestGlobalPosition.length);
                }

                // TODO: Die particle

                epoch++;
            }

        }
    }

    private void initParticles(){
        double hi = 0.1 * maxX;
        double lo = 0.1 * minX;
        double[] position = new double[neuralNetwork.getNumberOfWeights()];
        double[] velocity = new double[neuralNetwork.getNumberOfWeights()];
        double error = 0.0;
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

    private class Particle {
        private double[] position;
        private double error;
        private double[] velocity;
        private double[] bestPosition;
        private double bestError;

        private Particle(double[] position, double error, double[] velocity, double[] bestPosition, double bestError) {
            this.position = new double[position.length];
            this.velocity = new double[velocity.length];
            this.bestPosition = new double[bestPosition.length];

            System.arraycopy(this.position, 0, position, 0, position.length);
            this.error = error;
            System.arraycopy(this.velocity, 0, velocity, 0, velocity.length);
            System.arraycopy(this.bestPosition, 0, bestPosition, 0, bestPosition.length);
            this.bestError = bestError;
        }
    }
}
