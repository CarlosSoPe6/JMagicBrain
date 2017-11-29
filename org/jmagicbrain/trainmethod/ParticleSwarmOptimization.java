package org.jmagicbrain.trainmethod;

import org.jmagicbrain.functions.ErrorFunction;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.List;
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
    private final double socialGlobalConstant;
    private final double probDeath;

    public ParticleSwarmOptimization(double probDeath, double congitiveLocalConstant, double socialGlobalConstant, int numberOfParticles, double maxX, double minX, int maxEpochs, double maxError, List<List<Double>> trainingSet, List<List<Double>> expectedOutput, ErrorFunction errorFunction) {
        super(maxEpochs, maxError, trainingSet, expectedOutput, errorFunction);
        this.numberOfParticles = numberOfParticles;
        this.maxX = maxX;
        this.minX = minX;
        this.W = 0.7289;
        this.congitiveLocalConstant = congitiveLocalConstant;
        this.socialGlobalConstant = socialGlobalConstant;
        this.swarm = new Particle[this.numberOfParticles];
        this.bestGlobalError = Double.MAX_VALUE;
        this.probDeath = probDeath;

        this.random = new Random();
    }

    private void shuffle(int[] order){
        for (int i = 0; i < order.length; i++){
            int tmp = order[i];
            order[i] = order[random.nextInt(order.length)];
            order[random.nextInt(order.length)] = tmp;
        }
    }

    @Override
    public void train() {
        PrintWriter writer = null;
        try {
            writer = new PrintWriter("file.csv", "UTF-8");
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        StringBuilder sb = new StringBuilder();
        bestGlobalPosition = new double[neuralNetwork.getNumberOfWeights()];
        initParticles();
        int epoch = 0;
        double die = 0;
        double errt = 0;
        int[] sequence = new int[numberOfParticles];
        for (int i = 0; i < sequence.length; i++){
            sequence[i] = i;
        }
        double r1, r2;
        Particle currentParticle;
        double[] newPos = new double[neuralNetwork.getNumberOfWeights()];
        double[] newVel = new double[neuralNetwork.getNumberOfWeights()];

        while (epoch < maxEpochs){
            if(bestGlobalError < maxError) break;
            shuffle(sequence);
            for(int i = 0; i < sequence.length; i++){
                int ev = sequence[i];
                currentParticle = swarm[sequence[ev]];
                for(int j = 0; j < neuralNetwork.getNumberOfWeights(); j++){
                    r1 = random.nextDouble();
                    r2 = random.nextDouble();

                    newVel[j] =
                            (this.W * currentParticle.velocity[j]) +
                            (r1 * congitiveLocalConstant * (currentParticle.bestPosition[j] - currentParticle.position[j])) +
                            (r2 * socialGlobalConstant * (bestGlobalPosition[j] - currentParticle.position[j]));
                }

                for(int j = 0; j < neuralNetwork.getNumberOfWeights(); j++) {
                    newPos[j] = currentParticle.position[j] + newVel[i];
                    if(newPos[j] < minX) newPos[j] = minX;
                    else if(newPos[j] > maxX) newPos[j] = maxX;
                }

                System.arraycopy(newPos, 0, currentParticle.position, 0, currentParticle.position.length);
                System.arraycopy(newVel, 0, currentParticle.velocity, 0, currentParticle.velocity.length);

                neuralNetwork.setWeights(newPos);
                currentParticle.error = errorFunction.getError(trainingSet, expectedOutput);

                if(currentParticle.error < currentParticle.bestError){
                    currentParticle.bestError = currentParticle.error;
                    System.arraycopy(newPos, 0 ,currentParticle.bestPosition, 0, currentParticle.position.length);
                }

                if(currentParticle.error < bestGlobalError){
                    bestGlobalError = currentParticle.error;
                    System.arraycopy(newPos, 0, bestGlobalPosition, 0, bestGlobalPosition.length);
                }

                die = random.nextDouble();
                if(die < this.probDeath){
                    for(int k = 0; i < numberOfParticles; i++){
                        currentParticle.position[k] = (maxX - minX) * random.nextDouble() + minX;

                        neuralNetwork.setWeights(currentParticle.position);
                        currentParticle.error = errorFunction.getError(trainingSet, expectedOutput);
                        currentParticle.bestError = currentParticle.error;

                        if(currentParticle.error < bestGlobalError){
                            bestGlobalError = currentParticle.error;
                            System.arraycopy(currentParticle.position, 0, bestGlobalPosition, 0, bestGlobalPosition.length);
                        }
                    }
                }

                sb.append(String.format("%f\n", currentParticle.error));
            }
            epoch++;
        }

        System.out.println(Arrays.toString(bestGlobalPosition));
        System.out.printf("Best error: %f\n", bestGlobalError);

        writer.write(sb.toString());
        writer.close();

    }

    private void initParticles(){
        double hi = 0.05 * maxX;
        double lo = 0.05 * minX;
        double[] position = new double[neuralNetwork.getNumberOfWeights()];
        double[] velocity = new double[neuralNetwork.getNumberOfWeights()];
        double error = 0.0;
        for(int i = 0; i < numberOfParticles; i++){
            for(int j = 0; j < position.length; j++) {
                position[j] = (maxX - minX) * random.nextDouble() + minX;
                neuralNetwork.setWeights(position);
                error = super.errorFunction.getError(trainingSet, expectedOutput);
                velocity[j] = (hi- lo) * random.nextDouble() + lo;
            }
            swarm[i] = new Particle(position, error, velocity, position, error);
            if(error < bestGlobalError){
                bestGlobalError = error;
                System.arraycopy(position, 0, bestGlobalPosition, 0, bestGlobalPosition.length);
                System.out.println(Arrays.toString(bestGlobalPosition));
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
            this.position = position.clone();
            this.velocity = velocity.clone();
            this.bestPosition = bestPosition.clone();

            this.error = error;
            this.bestError = bestError;
        }
    }
}
