package org.jmagicbrain.utils;

import com.google.gson.Gson;
import org.jmagicbrain.NeuralNetwork;

import java.io.*;

/**
 * Neural Network input/output
 * Importa y exporta los valores de la red neuronal con archivos JSON
 */
public class NNIO {
    /**
     * Exporta los pesos de una red neuronal en un JSON
     * @param neuralNetwork Red neuronal que desea guardar
     * @param path Ruta donde se guarda el JSON
     */
    public static void exportNN (NeuralNetwork neuralNetwork, String path) throws IOException{
        //Crear el JSON con el API GSON
        MatWeights mw = new MatWeights(neuralNetwork.getWeights());
        Gson gson = new Gson();
        String json = gson.toJson(mw);

        //Abrir el archivo
        File f = new File(path);
        FileOutputStream fos = new FileOutputStream(f);
        ObjectOutputStream oos = new ObjectOutputStream(fos);

        //Escribir el JSON en el archivo seleccionado
        oos.writeObject(json);

        //Cerrar los archivos
        oos.close();
        fos.close();
    }

    /**
     * Importa los pesos de una red neuronal desde un archivo JSON
     * @param path Ruta del archivo JSON
     * @return double[][][] con los pesos
     */
    public static double[][][] importNN(String path) throws IOException{
        //Abrir el archivo seleccionado
        File f = new File(path);
        FileInputStream fis = new FileInputStream(f);
        ObjectInputStream ois = new ObjectInputStream(fis);

        String json;

        //Extraer el JSON
        try {
            json = (String)ois.readObject();
        }catch (Exception e) {
            json = "";
        }

        Gson gson = new Gson();
        //Guardar la instancia con los pesos del archivo
        MatWeights mw = gson.fromJson(json, MatWeights.class);

        return mw.weights;
    }

    /**
     * Importa los pesos de una red neuronal desde un archivo JSON,
     * y los inserta en la red neuronal
     * @param neuralNetwork Red neuronal a la que se le insertaran los pesos
     * @param path Ruta del archivo JSON
     * @return double[][][] con los pesos
     */
    public static double[][][] importNN(NeuralNetwork neuralNetwork, String path) throws IOException{
        //Abrir el archivo seleccionado
        File f = new File(path);
        FileInputStream fis = new FileInputStream(f);
        ObjectInputStream ois = new ObjectInputStream(fis);

        String json;

        //Extraer el JSON
        try {
            json = (String)ois.readObject();
        }catch (Exception e) {
            json = "";
        }

        Gson gson = new Gson();
        //Guardar la instancia con los pesos del archivo
        MatWeights mw = gson.fromJson(json, MatWeights.class);

        //Insertar los pesos en la red neuronal
        neuralNetwork.setWeights(mw.weights);

        return mw.weights;
    }

    /**
     * Clase auxiliar para hacer la conversion a JSON
     */
    private static class MatWeights {
        protected double[][][] weights;
        public MatWeights(double[][][] weights) {
            this.weights = weights;
        }
    }
}
