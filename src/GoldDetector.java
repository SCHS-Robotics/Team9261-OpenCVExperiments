import org.opencv.core.*;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

public class GoldDetector {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        //load input image and create output image
        Mat input = Imgcodecs.imread("C:\\TrainingArena\\positive_images\\IMG_1624.jpg");
        Mat output = input.clone();

        //color filtering
        Mat hsv = new Mat();
        //hmm... I guess this works somehow?
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_BGR2HSV);
        Imgproc.cvtColor(hsv, hsv, Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsv,new Scalar(23.886119773123486,0,0),new Scalar(37.16723549488056,255,255),hsv);
        showResult(hsv);

        //find edges
        Mat edges = new Mat();
        Imgproc.Canny(input,edges,0,255);
        //Mat edges2 = new Mat();
        //edges.copyTo(edges2,mask);
        //showResult(edges2); //output image

        FastFeatureDetector fast = FastFeatureDetector.create();
        MatOfKeyPoint kp = new MatOfKeyPoint();
        fast.detect(edges, kp);
        //Features2d.drawKeypoints(input,kp,output,new Scalar(0,255,0),0);
        //showResult(output);
    }
    private static void showResult(Mat display) {
        Mat img = display.clone();
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
