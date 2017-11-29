package org.jmagicbrain.trainmethod;

import org.jmagicbrain.functions.ErrorFunction;

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

    public ParticleSwarmOptimization(double probDeath, double w, double congitiveLocalConstant, double socialGlobalConstant, int numberOfParticles, double maxX, double minX, int maxEpochs, double maxError, List<List<Double>> trainingSet, List<List<Double>> expectedOutput, ErrorFunction errorFunction) {
        super(maxEpochs, maxError, trainingSet, expectedOutput, errorFunction);
        this.numberOfParticles = numberOfParticles;
        this.maxX = maxX;
        this.minX = minX;
        this.W = w;
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
        bestGlobalPosition = new double[neuralNetwork.getNumberOfWeights()];
        initParticles();
        int epoch = 0;
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

                neuralNetwork.setWeights(currentParticle.position);
                currentParticle.error = errorFunction.getError(trainingSet, expectedOutput);

                if(currentParticle.error < currentParticle.bestError){
                    currentParticle.bestError = currentParticle.error;
                    System.arraycopy(currentParticle.position, 0 ,currentParticle.bestPosition, 0, currentParticle.position.length);
                }

                if(currentParticle.error < bestGlobalError){
                    bestGlobalError = currentParticle.error;
                    System.arraycopy(currentParticle.position, 0, bestGlobalPosition, 0, bestGlobalPosition.length);
                }

                if(random.nextDouble() < this.probDeath){
                    for(int k = 0; i < numberOfParticles; i++){
                        currentParticle.position[k] = (maxX - minX) * random.nextDouble() + minX;
                    }
                    neuralNetwork.setWeights(currentParticle.position);
                    currentParticle.error = errorFunction.getError(trainingSet, expectedOutput);
                    currentParticle.bestError = currentParticle.error;

                    if(currentParticle.error < bestGlobalError){
                        bestGlobalError = currentParticle.error;
                        System.arraycopy(currentParticle.position, 0, bestGlobalPosition, 0, bestGlobalPosition.length);
                    }
                }
            }
            epoch++;
        }

        System.out.println(Arrays.toString(bestGlobalPosition));
        System.out.printf("Best error: %f\n", bestGlobalError);
        printDistances();
    }

    private void initParticles(){
        double hi = 0.095 * maxX;
        double lo = 0.095 * minX;
        double[] position = new double[neuralNetwork.getNumberOfWeights()];
        double[] velocity = new double[neuralNetwork.getNumberOfWeights()];
        double error = 0.0;
        for(int i = 0; i < numberOfParticles; i++){
            for(int j = 0; j < position.length; j++) {
                position[j] = (maxX - minX) * random.nextDouble() + minX;
                velocity[j] = (hi- lo) * random.nextDouble() + lo;
            }
            swarm[i] = new Particle(position, error, velocity, position, error);
            neuralNetwork.setWeights(swarm[i].position);
            error = super.errorFunction.getError(trainingSet, expectedOutput);
            if(error < bestGlobalError){
                bestGlobalError = error;
                System.arraycopy(position, 0, bestGlobalPosition, 0, bestGlobalPosition.length);
                System.out.println(Arrays.toString(bestGlobalPosition));
            }
        }


        swarm[0].position = new double[] {
                    -3.0844564959493583,
                    -1.2967465354470171,
                    -3.4568230867127125,
                    2.427439016134647,
                    -0.20166162078097105,
                    -0.4290972662466771,
                    -0.4697318183992755,
                    0.6120798051855239,
                    -1.2037172045033857,
                    -2.6851589882808415,
                    2.511666622784259,
                    -2.166286047950086,
                    -3.218021289916955,
                    2.7437658341517515,
                    -5.936634301991448,
                    -2.5146529219198013,
                    -6.666062815913475,
                    2.668143796200943,
                    -1.148340328571109,
                    -0.3666399913890762,
                    0.061704049785499704,
                    1.7943803457634186,
                    -1.1354221107143903,
                    -6.720335520866052,
                    6.475759097030569,
                    -1.093673628747236,
                    -5.908701589833334,
                    4.646373751207232,
                    1.8761012387864753,
                    0.6907004154874573,
                    2.028270979396172,
                    -2.302496472525233,
                    -0.42202901840671436,
                    -0.20374994356740514,
                    1.0815174678404853,
                    -1.3649705635082576,
                    0.6111084428476541,
                    1.2242951174694339,
                    -2.6361997551615364,
                    1.4976621273579531,
                    3.515643547084648,
                    -3.2995779551140916,
                    1.0038961361132481,
                    0.3267767713778661,
                    1.0410957890092047,
                    -1.879280383352821,
                    -0.5130589321833102,
                    0.565484633783712,
                    1.2702766548623183,
                    -0.7707257886857778,
                    0.5612085472740267,
                    0.24176654837651126,
                    -0.854833276051476,
                    1.7138438689264544,
                    2.18359059945286,
                    -2.0666449511989984,
                    -7.917746367092543,
                    -9.258325648114015,
                    7.67840062368662,
                    6.987379267150389,
                    -7.159177173676007,
                    -4.298621009310246,
                    14.430268876230658,
                    -11.35862286210741,
                    8.825803282410579,
                    -2.3367301440658004,
                    1.8959206322885984,
                    -13.391921171829358,
                    3.0972128133949615,
                    -13.079045310684466,
                    5.02949611890173
        };

        neuralNetwork.setWeights(swarm[0].position);
        error = super.errorFunction.getError(trainingSet, expectedOutput);
        if(error < bestGlobalError){
            bestGlobalError = error;
            System.arraycopy(swarm[0].position, 0, bestGlobalPosition, 0, bestGlobalPosition.length);
            System.out.println(Arrays.toString(bestGlobalPosition));
        }

        System.out.println(bestGlobalError);
        System.out.println(error);

        swarm[0].error = error;
        swarm[0].bestError = error;
        swarm[0].bestPosition = swarm[0].position.clone();


    }

    private void printDistances(){
        double sum;
        for(int i = 0; i < swarm.length; i++){
            sum = 0;
            for(int j = 0; j < bestGlobalPosition.length; j++){
                sum += (bestGlobalPosition[j] - swarm[i].position[j]) * (bestGlobalPosition[j] - swarm[i].position[j]);
            }
            sum = Math.sqrt(sum);
            System.out.printf("Particle: %d, %f\n", i+1, sum);
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
