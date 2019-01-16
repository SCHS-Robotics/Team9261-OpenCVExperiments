import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class HoughCircle {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        //20180910_095634
        //20180910_094912
        Mat input = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\20180910_094912.jpg");

        List<Mat> channels = new ArrayList<>();
        Core.split(input,channels);

        Mat mask = new Mat();

        Core.bitwise_and(channels.get(0),channels.get(1),mask);
        Core.bitwise_and(mask,channels.get(2),mask);

        Imgproc.GaussianBlur(mask,mask,new Size(15,15),0);

        Imgproc.threshold(mask,mask,0,255,Imgproc.THRESH_OTSU);

        showResult(mask);

        Mat circles = new Mat();

        Mat edges = new Mat();
        Imgproc.Canny(mask,edges,0,255);

        Imgproc.dilate(edges,edges,Imgproc.getStructuringElement(Imgproc.MORPH_CROSS,new Size(5,5)));

        List<MatOfPoint> contours = new ArrayList<>();

        Imgproc.findContours(edges,contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        for(MatOfPoint c : contours) {
            double areaCircularity = Imgproc.contourArea(c)/ Math.pow(Math.PI*(Imgproc.boundingRect(c).width/2),2);
            double perimeterCircularity = Imgproc.arcLength(new MatOfPoint2f(c.toArray()),true)/(Math.PI*Imgproc.boundingRect(c).width);
            if(Math.abs(1-areaCircularity) <= 1 && Math.abs(1-perimeterCircularity) <= 1 && Imgproc.contourArea(c) >= 1000) {
                Imgproc.drawContours(input,contours,contours.indexOf(c),new Scalar(0,255,0),5);
                Imgproc.putText(input,"Area Circularity: "+ Double.toString(Math.floor(100*areaCircularity)/100.0),new Point(Imgproc.boundingRect(c).x,Imgproc.boundingRect(c).y),Imgproc.FONT_HERSHEY_PLAIN,5,new Scalar(255,255,255),3);
            }
        }
        showResult(input);
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
