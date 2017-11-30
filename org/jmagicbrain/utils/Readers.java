package org.jmagicbrain.utils;

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

}
