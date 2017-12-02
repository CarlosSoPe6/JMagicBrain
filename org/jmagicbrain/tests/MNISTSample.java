package org.jmagicbrain.tests;

import org.jmagicbrain.NeuralNetwork;
import org.jmagicbrain.functions.ActivationFunction;
import org.jmagicbrain.functions.ErrorFunction;
import org.jmagicbrain.functions.MeanSquaredError;
import org.jmagicbrain.functions.Sigmoid;
import org.jmagicbrain.initializers.DefaultInitializer;
import org.jmagicbrain.initializers.WeightInitializer;
import org.jmagicbrain.trainmethod.BackPropagation;
import org.jmagicbrain.trainmethod.TrainMethod;
import org.jmagicbrain.utils.Normalizers;
import org.jmagicbrain.utils.Readers;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class MNISTSample extends JFrame {

    private JLabel asnwerlabel;
    private JButton imageButton;
    private JLabel jLabel1;
    private JPanel rootPanel;
    private JPanel imagePanel;

    private double[][] images;
    private double[][] expected;

    private ActivationFunction activationFunction;
    private ErrorFunction errorFunction;
    private WeightInitializer weightInitializer;
    private TrainMethod trainMethod;
    private NeuralNetwork neuralNetwork;

    public MNISTSample() throws IOException {
        super("MNIST Sample");

        initUI();

        initNeuralNet();

        super.setVisible(true);
    }

    private void initUI(){
        rootPanel = new JPanel();
        imageButton = new JButton();
        jLabel1 = new JLabel();
        asnwerlabel = new JLabel();
        imagePanel = new  JPanel();

        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setBackground(new java.awt.Color(255, 255, 255));
        setMinimumSize(new java.awt.Dimension(254, 254));
        setName("rootFrame"); // NOI18N
        setPreferredSize(new java.awt.Dimension(254, 254));
        setResizable(false);
        setSize(new java.awt.Dimension(254, 254));

        rootPanel.setBackground(new java.awt.Color(255, 255, 255));
        rootPanel.setPreferredSize(new java.awt.Dimension(254, 254));

        imageButton.setBackground(new java.awt.Color(255, 255, 255));
        imageButton.setText("New image");
        imageButton.addActionListener(this::imageButtonActionPerformed);

        jLabel1.setBackground(new java.awt.Color(255, 255, 255));
        jLabel1.setForeground(new java.awt.Color(0, 0, 0));
        jLabel1.setText("La imagen es");
        jLabel1.setToolTipText("");

        asnwerlabel.setBackground(new java.awt.Color(255, 255, 255));
        asnwerlabel.setForeground(new java.awt.Color(0, 0, 0));
        asnwerlabel.setText("X");

        imagePanel.setPreferredSize(new java.awt.Dimension(28, 28));

        GroupLayout jPanel2Layout = new GroupLayout(imagePanel);
        imagePanel.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
                jPanel2Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                        .addGap(0, 192, Short.MAX_VALUE)
        );
        jPanel2Layout.setVerticalGroup(
                jPanel2Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                        .addGap(0, 192, Short.MAX_VALUE)
        );

        GroupLayout jPanel1Layout = new GroupLayout(rootPanel);
        rootPanel.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
                jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                        .addGroup(jPanel1Layout.createSequentialGroup()
                                .addGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.TRAILING, false)
                                        .addGroup(GroupLayout.Alignment.LEADING, jPanel1Layout.createSequentialGroup()
                                                .addContainerGap()
                                                .addComponent(imagePanel, GroupLayout.DEFAULT_SIZE, 192, Short.MAX_VALUE))
                                        .addGroup(GroupLayout.Alignment.LEADING, jPanel1Layout.createSequentialGroup()
                                                .addComponent(imageButton)
                                                .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
                                                .addComponent(jLabel1)))
                                .addPreferredGap(LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(asnwerlabel, GroupLayout.DEFAULT_SIZE, 50, Short.MAX_VALUE)
                                .addContainerGap())
        );
        jPanel1Layout.setVerticalGroup(
                jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                        .addGroup(GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                                .addContainerGap()
                                .addComponent(imagePanel, GroupLayout.PREFERRED_SIZE, 192, GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED, 36, Short.MAX_VALUE)
                                .addGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING, false)
                                        .addComponent(imageButton)
                                        .addGroup(GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                                                .addGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                                                        .addComponent(jLabel1, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                                        .addComponent(asnwerlabel, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                                                .addContainerGap())))
        );

        GroupLayout layout = new GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
                layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                        .addComponent(rootPanel, GroupLayout.DEFAULT_SIZE, 266, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
                layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                        .addComponent(rootPanel, GroupLayout.DEFAULT_SIZE, 266, Short.MAX_VALUE)
        );

        pack();
    }

    private void initNeuralNet() throws IOException {
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
    }

    private void imageButtonActionPerformed(ActionEvent e) {
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

            asnwerlabel.setText(maxIndex + "");

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
            if(f.isFile() && i < 100){
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
