package org.jmagicbrain.utils;

import org.jetbrains.annotations.Contract;

public class Normalizers {
    @Contract(pure = true)
    public static double normalizeRange(double value, double min, double max){
        return (value - min) / (max - min);
    }

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
}
