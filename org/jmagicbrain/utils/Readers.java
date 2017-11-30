package org.jmagicbrain.utils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;

public class Readers {

    public static byte[] readFileAsBytes(String path) throws IOException {
        byte[] b;
        File file = new File(path);
        b = new byte[(int) file.length()];
        InputStream is = new FileInputStream(file);
        is.read(b);
        is.close();

        return b;
    }

    public static String reafFileAsString(String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        BufferedReader reader = new BufferedReader(new FileReader (path));
        String line;
        while ((line = reader.readLine()) != null){
            sb.append(line);
        }

        return sb.toString();
    }

    public static int[] getBufferedImage(String path) throws IOException {
        File f = new File(path);
        BufferedImage img = ImageIO.read(f);
        int[] data = new int[img.getHeight() * img.getWidth()];
        int count = 0;
        for(int x = 0; x < img.getHeight(); x++) {
            for (int y = 0; y < img.getWidth(); y++) {
                data[count] = img.getRGB(x, y);
                count++;
            }
        }

        return data;
    }

}
