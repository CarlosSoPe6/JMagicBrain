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
import org.jmagicbrain.utils.NNIO;
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

    private JLabel asnwerLabel;
    private JButton imageButton;
    private JLabel jLabel1;
    private JPanel rootPanel;
    private JLabel imagePanel;

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
        asnwerLabel = new JLabel();
        imagePanel = new JLabel();

        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setBackground(new Color(255, 255, 255));
        setMinimumSize(new Dimension(254, 254));
        setName("rootFrame"); // NOI18N
        setPreferredSize(new Dimension(254, 254));
        setResizable(false);
        setSize(new Dimension(254, 254));

        rootPanel.setBackground(new Color(255, 255, 255));
        rootPanel.setPreferredSize(new Dimension(254, 254));

        imageButton.setBackground(new Color(255, 255, 255));
        imageButton.setText("New image");
        imageButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                imageButtonActionPerformed(evt);
            }
        });

        jLabel1.setBackground(new Color(255, 255, 255));
        jLabel1.setForeground(new Color(0, 0, 0));
        jLabel1.setText("La imagen es");
        jLabel1.setToolTipText("");

        asnwerLabel.setBackground(new Color(255, 255, 255));
        asnwerLabel.setForeground(new Color(0, 0, 0));
        asnwerLabel.setText("X");

        imagePanel.setMaximumSize(new Dimension(64, 64));
        imagePanel.setMinimumSize(new Dimension(64, 64));
        imagePanel.setPreferredSize(new Dimension(64, 64));

        GroupLayout jPanel1Layout = new GroupLayout(rootPanel);
        rootPanel.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
                jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                        .addGroup(jPanel1Layout.createSequentialGroup()
                                .addGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.TRAILING, false)
                                        .addGroup(GroupLayout.Alignment.LEADING, jPanel1Layout.createSequentialGroup()
                                                .addContainerGap()
                                                .addComponent(imagePanel, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                                        .addGroup(GroupLayout.Alignment.LEADING, jPanel1Layout.createSequentialGroup()
                                                .addComponent(imageButton)
                                                .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
                                                .addComponent(jLabel1)))
                                .addGap(35, 35, 35)
                                .addComponent(asnwerLabel, GroupLayout.DEFAULT_SIZE, 27, Short.MAX_VALUE)
                                .addContainerGap())
        );
        jPanel1Layout.setVerticalGroup(
                jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                        .addGroup(GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                                .addContainerGap()
                                .addComponent(imagePanel, GroupLayout.DEFAULT_SIZE, 210, Short.MAX_VALUE)
                                .addGap(18, 18, 18)
                                .addGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING, false)
                                        .addComponent(imageButton)
                                        .addGroup(GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                                                .addGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                                                        .addComponent(jLabel1, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                                        .addComponent(asnwerLabel, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
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

        trainMethod = new BackPropagation.BackPropagationBuilder()
                .setErrorFunction(errorFunction)
                .setMaxEpochs(5000)
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
                .addLayer(16)
                .addLayer(10)
                .build();

        System.out.println("Starting train");
        System.out.println(neuralNetwork.train());
        System.out.println("Ending train");

        NNIO.exportNN(neuralNetwork, "C:\\\\Desarrollo\\\\DataSets\\\\Numeros\\f1000e5000.json");

        //NNIO.importNN(neuralNetwork, "C:\\\\Desarrollo\\\\DataSets\\\\Numeros\\f100e1000.json");
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
            imagePanel.setIcon(new ImageIcon(bufferedImage));
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

            asnwerLabel.setText(maxIndex + "");

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
            if(f.isFile() && i < 1000){
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
