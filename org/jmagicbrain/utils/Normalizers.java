package org.jmagicbrain.utils;

import org.jetbrains.annotations.Contract;

public class Normalizers {
    @Contract(pure = true)
    public static double normalizeRange(double value, double min, double max){
        return (value - min) / (max - min);
    }
}
