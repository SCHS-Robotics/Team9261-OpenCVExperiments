import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class gold {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String[] args) {
        Mat testImage = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\20180910_095634.jpg");
        Mat x = new Mat();
        Mat y = new Mat();
        Mat grad = new Mat();
        Mat bw = new Mat();
        Imgproc.cvtColor(testImage,bw,Imgproc.COLOR_BGR2GRAY);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(7,7));

        Imgproc.Sobel(bw,x,3,1,0);
        Imgproc.Sobel(bw,y,3,0,1);

        Mat x2 = new Mat();
        Mat y2 = new Mat();

        Mat sum = new Mat();

        Core.multiply(x,x,x2);
        Core.multiply(y,y,y2);

        Core.add(x2,y2,sum);

        sum.convertTo(sum,CvType.CV_32F);

        Core.pow(sum,0.5,grad);

        Mat edges = new Mat();

        grad.convertTo(grad,CvType.CV_8UC1);

        //Imgproc.Canny(testImage,edges,100,120);

        //Core.add(edges,grad,edges);

        Imgproc.threshold(grad,edges,50,255,Imgproc.THRESH_BINARY);

        List<MatOfPoint> contours = new ArrayList<>();

        Imgproc.dilate(edges,edges,kernel);
        //Imgproc.erode(edges,edges,kernel);

        Imgproc.findContours(edges,contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
        List<MatOfPoint> approxList = new ArrayList<>();
        for(int i = 0; i < contours.size(); i++) {
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(new MatOfPoint2f(contours.get(i).toArray()),approx,0.01*Imgproc.arcLength(new MatOfPoint2f(contours.get(i).toArray()),true),true);
            approxList.add(new MatOfPoint(approx.toArray()));
            if(approx.size().height >= 4 && Imgproc.contourArea(contours.get(i)) > 1000) {
                Rect bbox = Imgproc.boundingRect(new MatOfPoint(approx.toArray()));
                Mat input = bw.submat(bbox);
                int std = (int) Math.round(calcStdDev(input));
                Moments moments = Imgproc.moments(approx);
                Imgproc.putText(testImage, Integer.toString(std),new Point((int)(moments.m10/moments.m00),(int) (moments.m01/moments.m00)), Imgproc.FONT_HERSHEY_COMPLEX,3,new Scalar(0,0,255),3);
                Imgproc.drawContours(testImage,approxList,i,new Scalar(0,255,0),5);

            }
        }


        showResult(edges);
        showResult(testImage);
    }

    private static double calcStdDev(Mat in) {
        MatOfDouble std = new MatOfDouble();
        Core.meanStdDev(in,new MatOfDouble(),std);
        System.out.println(std.dump());
        return std.get(0,0)[0];
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
