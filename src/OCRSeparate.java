import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;

public class OCRSeparate {

    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}

    public static void main(String args[]) {
        String inputPath = "C:\\TrainingArena\\OCR\\digits.png";
        Mat input = Imgcodecs.imread(inputPath,0);
        System.out.println(input.cols());
        System.out.println(input.rows());
        int z = 0;
        int prevx = 0;
        int prevy = 0;
        for (int i = 0; i < 1000; i += 20) { //rows
            for(int j =0; j < 2000; j += 20) { //cols
                z++;
                System.out.println(i+", "+(i+20)+", "+j+", "+(j+20));
                Mat b = input.submat(i,i+20,j,j+20);
                Imgcodecs.imwrite("C:\\TrainingArena\\OCR\\Training\\"+z+".png",b);
            }
        }
        showResult(input);
    }
    private static void showResult(Mat img) {
        Imgproc.resize(img, img, new Size(640, 480));
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", img, matOfByte);
        byte[] byteArray = matOfByte.toArray();
        BufferedImage bufImage = null;
        try {
            InputStream in = new ByteArrayInputStream(byteArray);
            bufImage = ImageIO.read(in);
            JFrame frame = new JFrame();
            frame.getContentPane().add(new JLabel(new ImageIcon(bufImage)));
            frame.pack();
            frame.setVisible(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
