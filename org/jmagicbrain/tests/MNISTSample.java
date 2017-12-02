package org.jmagicbrain.tests;

import org.jmagicbrain.NeuralNetwork;
import org.jmagicbrain.functions.ActivationFunction;
import org.jmagicbrain.functions.ErrorFunction;
import org.jmagicbrain.functions.MeanSquaredError;
import org.jmagicbrain.functions.Sigmoid;
import org.jmagicbrain.initializers.DefaultInitializer;
import org.jmagicbrain.initializers.WeightInitializer;
import org.jmagicbrain.trainmethod.BackPropagation;
import org.jmagicbrain.trainmethod.ParticleSwarmOptimization;
import org.jmagicbrain.trainmethod.TrainMethod;
import org.jmagicbrain.utils.Normalizers;
import org.jmagicbrain.utils.Readers;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class MNISTSample extends JFrame {
    private JPanel rootPanel;
    private JButton loadNewImageButton;
    private JPanel imagePanel;
    private JLabel predictedName;

    private double[][] images;
    private double[][] expected;

    private final ActivationFunction activationFunction;
    private final ErrorFunction errorFunction;
    private final WeightInitializer weightInitializer;
    private final TrainMethod trainMethod;
    private final NeuralNetwork neuralNetwork;

    public MNISTSample() throws IOException {
        super("MNIST");
        super.setContentPane(rootPanel);
        super.pack();
        super.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        ArrayList<File> files = new ArrayList<>();
        getFiles("C:\\Desarrollo\\DataSets\\Numeros\\Train", files);
        images = new double[files.size()][784];
        expected = new double[files.size()][10];
        int[] image;

        for(int i = 0; i < files.size(); i++){
            expected[i][files.get(i).getParent().charAt(files.get(i).getParent().length() - 1) - '0'] = 1.0;

            image = Readers.getBufferedImage(files.get(i));

            for(int j = 0; j < image.length; j++){
                images[i][j] = Normalizers.normalizeRange((((image[j] >> 16) & 0xFF) * 0.3) + (((image[j] >> 8) & 0xFF) *0.59) + ((image[j] & 0xFF) * 0.11 ), 0.0, 255.0);
            }
        }

        System.gc();

        activationFunction = new Sigmoid();
        errorFunction = new MeanSquaredError();
        weightInitializer = new DefaultInitializer();

        /*trainMethod = new ParticleSwarmOptimization.ParticleSwarmOptimizationBuilder()
                .setProbDeath(0.09)
                .setW(0.10936)
                .setCongitiveLocalConstant(0.09301)
                .setSocialGlobalConstant(0.09301)
                .setNumberOfParticles(10000)
                .setMaxX(60.0)
                .setMinX(-60.0)
                .setMaxEpochs(1000)
                .setMaxError(0.1)
                .setTrainingSet(images)
                .setExpectedOutput(expected)
                .setErrorFunction(errorFunction)
                .build();*/

        trainMethod = new BackPropagation.BackPropagationBuilder()
                .setErrorFunction(errorFunction)
                .setMaxEpochs(20000)
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
                //.addLayer(16)
                .addLayer(16)
                .addLayer(10)
                .build();

        System.out.println("Starting train");
        System.out.println(neuralNetwork.train());
        System.out.println("Ending train");



        super.setVisible(true);
        loadNewImageButton.addActionListener(this::onLoadNewImageActionPerformed);
    }

    private void onLoadNewImageActionPerformed(ActionEvent e) {
        FileDialog fd = new FileDialog(this, "Choose a file", FileDialog.LOAD);
        fd.setDirectory("C:\\");
        fd.setVisible(true);
        File file = fd.getFiles()[0];

        int[] image;
        double[] inputData;
        double max = Double.MIN_VALUE;
        int maxIndex = 0;
        try {
            image = Readers.getBufferedImage(file);
            BufferedImage bufferedImage = ImageIO.read(file);
            imagePanel.paint(bufferedImage.getGraphics());
            inputData = new double[image.length];
            for(int j = 0; j < image.length; j++){
                inputData[j] = Normalizers.normalizeRange((((image[j] >> 16) & 0xFF) * 0.3) + (((image[j] >> 8) & 0xFF) *0.59) + ((image[j] & 0xFF) * 0.11 ), 0.0, 255.0);
            }
            neuralNetwork.setInputLayer(inputData);
            neuralNetwork.think();

            System.out.println(Arrays.toString(neuralNetwork.getOutputLayer()));

            for(int i = 0; i < neuralNetwork.getOutputLayer().length - 1; i++){
                if(neuralNetwork.getOutputLayer()[i] > max){
                    maxIndex = i;
                    max = neuralNetwork.getOutputLayer()[i];
                }
            }

        } catch (IOException e1) {
            e1.printStackTrace();
            JOptionPane.showMessageDialog(null, "Error de lectura");
        }
    }

    private void getFiles(String dirName, ArrayList<File> files){
        File dir = new File(dirName);
        File[] fileList = dir.listFiles();

        if(fileList == null) return;

        int i = 0;
        for(File f : fileList){
            if(f.isFile() && i < 70){
                files.add(f);
                i++;
            }else if(f.isDirectory()){
                getFiles(f.getAbsolutePath(), files);
            }
        }
    }

    public static void main(String[] args) throws IOException {
        MNISTSample sample = new MNISTSample();
    }
}
