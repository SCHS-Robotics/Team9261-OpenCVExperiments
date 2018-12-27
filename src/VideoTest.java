import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

public class VideoTest {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        Mat mat = new Mat();

        VideoCapture cap;
        int width = 640, height = 480;
        cap = new VideoCapture(0);
        if (!cap.isOpened()) {
            System.out.println("Camera Error");
        } else {
            System.out.println("Camera OK?");
            cap.set(Videoio.CAP_PROP_FRAME_WIDTH, width);
            cap.set(Videoio.CAP_PROP_FRAME_HEIGHT, height);
        }
        try {
            Thread.sleep(1000);
        } catch (InterruptedException ex) {
        }
        cap.read(mat);
        System.out.println("width, height = " + mat.cols() + ", " + mat.rows());
        while(true) {
            cap.read(mat);
            showResult(mat);
        }
    }
    private static void showResult(Mat display) {
        Mat img = display.clone();
        Imgproc.resize(img, img, new Size(640, (int) Math.round((640/img.size().width)*img.size().height)));
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", img, matOfByte);
        byte[] byteArray = matOfByte.toArray();
        BufferedImage bufImage;
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

