package org.jmagicbrain.utils;

public class Normalizers {

    /**
     * Entrega un valor de 0 a 1
     * @param value Valor a normalizar
     * @param min Valor mínimo posible
     * @param max Valor máximo posible
     * @return 'value' normalizado de 0 a 1
     */
    public static double normalizeRange(double value, double min, double max){
        return (value - min) / (max - min);
    }

    /**
     * Normaliza una matriz de Character
     * @param value La matriz a normalzar
     * @return Matriz de doubles normalizada
     */
    public static double[][] normalize(char[][] value){
        char[] mins = new char[value.length];
        char[] maxs = new char[value.length];

        for(int i = 0; i < value.length; i++){
            for(int j = 0; j < value[i].length; j++){
                if(mins[j] > value[i][j]) mins[j] = value[i][j];
                if(maxs[j] < value[i][j]) maxs[j] = value[i][j];
            }
        }

        double[][] matrix = new double[value.length][value[0].length];

        for(int i = 0; i < value.length; i++){
            for(int j = 0; j < value[i].length; j++){
                matrix[i][j] = normalizeRange(value[i][j], mins[j], maxs[j]);
            }
        }

        return matrix;
    }

    /**
     * Normaliza una matriz de Byte
     * @param value La matriz rectangular a normalizar
     * @return Matriz de doubles normalizada
     */
    public static double[][] normalize(byte[][] value){
        byte[] mins = new byte[value.length];
        byte[] maxs = new byte[value.length];

        for(int i = 0; i < value.length; i++){
            for(int j = 0; j < value[i].length; j++){
                if(mins[j] > value[i][j]) mins[j] = value[i][j];
                if(maxs[j] < value[i][j]) maxs[j] = value[i][j];
            }
        }

        double[][] matrix = new double[value.length][value[0].length];

        for(int i = 0; i < value.length; i++){
            for(int j = 0; j < value[i].length; j++){
                matrix[i][j] = normalizeRange(value[i][j], mins[j], maxs[j]);
            }
        }

        return matrix;
    }

    /**
     * Normaliza una matriz de Byte tomando normalizando 0x00 a 0 y 0xFF a 1
     * @param value La matriz rectangular a normalizar
     * @return Matriz de doubles normalizada
     */
    public static double[][] normalizeWithDefaults(byte[][] value){
        double[][] matrix = new double[value.length][value[0].length];

        for(int i = 0; i < value.length; i++){
            for(int j = 0; j < value[i].length; j++){
                matrix[i][j] = normalizeRange(value[i][j], Byte.MIN_VALUE, Byte.MAX_VALUE);
            }
        }

        return matrix;
    }

    /**
     * Normaliza una matriz de Integer tomando normalizando 0x00 a 0 y 0xFF a 1
     * @param value La matriz rectangular a normalizar
     * @return Matriz de doubles normalizada
     */
    public static double[][] normalizeWithDefaults(int[][] value){
        double[][] matrix = new double[value.length][value[0].length];

        for(int i = 0; i < value.length; i++){
            for(int j = 0; j < value[i].length; j++){
                matrix[i][j] = normalizeRange(value[i][j], Integer.MIN_VALUE, Integer.MAX_VALUE);
            }
        }

        return matrix;
    }

    public static double[] normalize(byte[] value){
        double[] data = new double[value.length];
        for(int j = 0; j < value.length; j++){
            data[j] = normalizeRange(value[j], Byte.MIN_VALUE, Byte.MAX_VALUE);
        }

        return data;
    }
}
