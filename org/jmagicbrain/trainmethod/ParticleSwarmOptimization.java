package org.jmagicbrain.trainmethod;

import org.jmagicbrain.exceptions.InvalidTrainingMethodArguments;
import org.jmagicbrain.functions.ErrorFunction;

import java.util.Arrays;
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

    private ParticleSwarmOptimization(double probDeath, double w, double congitiveLocalConstant, double socialGlobalConstant, int numberOfParticles, double maxX, double minX, int maxEpochs, double maxError, double[][] trainingSet, double[][] expectedOutput, ErrorFunction errorFunction) {
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

    /**
     * Da un orden aleatoreo a un arreglo
     * @param order arreglo a revolver
     */
    private void shuffle(int[] order){
        for (int i = 0; i < order.length; i++){
            int tmp = order[i];
            order[i] = order[random.nextInt(order.length)];
            order[random.nextInt(order.length)] = tmp;
        }
    }

    @Override
    public double train() {
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
                    newPos[j] = currentParticle.position[j] + newVel[j];
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

        return bestGlobalError;
    }

    /**
     * Inicializa las partículas
     */
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

    /**
     * Constructor para 'ParticleSwarmOptimization'
     */
    public static class ParticleSwarmOptimizationBuilder {
        private double probDeath;
        private double w;
        private double congitiveLocalConstant;
        private double socialGlobalConstant;
        private int numberOfParticles;
        private double maxX;
        private double minX;
        private int maxEpochs;
        private double maxError;
        private double[][] trainingSet;
        private double[][] expectedOutput;
        private ErrorFunction errorFunction;

        /**
         * Establece la probabilidad de muerte de una partícula
         * @param probDeath La probablilidad de merte
         * @return Referencia a si mismo
         */
        public ParticleSwarmOptimizationBuilder setProbDeath(double probDeath) {
            this.probDeath = probDeath;
            return this;
        }

        /**
         * Establece la importancia que se le drá a la velocidad anterior
         * @param w El valor de la constane
         * @return Referencia a si mismo
         */
        public ParticleSwarmOptimizationBuilder setW(double w) {
            this.w = w;
            return this;
        }

        /**
         * Establece el valor de la constante 'cognitive local', la cual le da la importancia a el mejor valor del
         * enjambre  al momento de moverse
         * @param congitiveLocalConstant El valor de la constante
         * @return Referencia a si mismo
         */
        public ParticleSwarmOptimizationBuilder setCongitiveLocalConstant(double congitiveLocalConstant) {
            this.congitiveLocalConstant = congitiveLocalConstant;
            return this;
        }

        /**
         * Establece el valor de la constante 'social global', la cual le da la importancia a el mejor valor del
         * enjambre al momento de moverse
         * @param socialGlobalConstant El valor de la constante
         * @return Referencia a si mismo
         */
        public ParticleSwarmOptimizationBuilder setSocialGlobalConstant(double socialGlobalConstant) {
            this.socialGlobalConstant = socialGlobalConstant;
            return this;
        }

        /**
         * Establece el númoero de partículas
         * @param numberOfParticles El número de particulas
         * @return Referencia a si mismo
         */
        public ParticleSwarmOptimizationBuilder setNumberOfParticles(int numberOfParticles) {
            this.numberOfParticles = numberOfParticles;
            return this;
        }

        /**
         * Establece el valo máximo de los pesos
         * @param maxX El valor máximo de los pesos
         * @return Referencia a si mismo
         */
        public ParticleSwarmOptimizationBuilder setMaxX(double maxX) {
            this.maxX = maxX;
            return this;
        }

        /**
         * Establece el valor mínimo de los pesos
         * @param minX El valor mínimo de los pesos
         * @return Referencia a si mismo
         */
        public ParticleSwarmOptimizationBuilder setMinX(double minX) {
            this.minX = minX;
            return this;
        }

        /**
         * Establece las epocas máximas
         * @param maxEpochs Las epocas máximas
         * @return Referencia a si mismo
         */
        public ParticleSwarmOptimizationBuilder setMaxEpochs(int maxEpochs) {
            this.maxEpochs = maxEpochs;
            return this;
        }

        /**
         * Establece el error para la condición de paro
         * @param maxError Error esperado
         * @return Referencia a si mismo
         */
        public ParticleSwarmOptimizationBuilder setMaxError(double maxError) {
            this.maxError = maxError;
            return this;
        }

        /**
         * Establece los datos de entrenamiento
         * @param trainingSet Lista de los datos de entrenamiento
         * @return Referencia a si mismo
         */
        public ParticleSwarmOptimizationBuilder setTrainingSet(double[][] trainingSet) {
            this.trainingSet = trainingSet;
            return this;
        }

        /**
         * Establece los datos esperados
         * @param expectedOutput Lista de los datos esperados
         * @return Referencia a si mismo
         */
        public ParticleSwarmOptimizationBuilder setExpectedOutput(double[][] expectedOutput) {
            this.expectedOutput = expectedOutput;
            return this;
        }

        /**
         * Establece la función de error
         * @param errorFunction Instancia de ErrorFunction
         * @return Referencia a si mismo
         */
        public ParticleSwarmOptimizationBuilder setErrorFunction(ErrorFunction errorFunction) {
            this.errorFunction = errorFunction;
            return this;
        }

        /**
         * Crea una instancia de ParticleSwarmOptimization con los parámetros dados
         * @return Instacia de ParticleSwarmOptimization
         * @throws InvalidTrainingMethodArguments En caso de tener algún argumento inválido
         */
        public ParticleSwarmOptimization build() throws InvalidTrainingMethodArguments{

            // TODO: Check nulls

            return new ParticleSwarmOptimization(
                probDeath,
                w,
                congitiveLocalConstant,
                socialGlobalConstant,
                numberOfParticles,
                maxX,
                minX,
                maxEpochs,
                maxError,
                trainingSet,
                expectedOutput,
                errorFunction
            );
        }
    }
}
