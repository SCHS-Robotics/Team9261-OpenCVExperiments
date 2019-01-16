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

public class Test5 {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    //static CascadeClassifier hexagons = new CascadeClassifier("C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Haar\\Training Arena\\classifier\\cascade.xml");
    public static void main(String args[]) {
        Mat raw = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Test Images\\occlusion.jpg",Imgcodecs.IMREAD_GRAYSCALE);
        Mat out = new Mat();
        //Imgproc.resize(raw,raw,new Size(640,360),0,0,Imgproc.INTER_AREA);
        Imgproc.cvtColor(raw,out,Imgproc.COLOR_GRAY2BGR);
        Imgproc.GaussianBlur(raw,raw,new Size(7,7),0);
        Mat canny = auto_canny(raw);
        List<MatOfPoint> contours = new ArrayList<>();
        List<MatOfPoint> hullPoints = new ArrayList<>();
        Imgproc.findContours(canny,contours,new Mat(),Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);
        for(MatOfPoint c : contours) {
            MatOfPoint hullPoint;
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(c, hull);
            hullPoint = convertIndexesToPoints(c,hull); //Finds the MatOfPoint that the hull describes
            hullPoints.add(hullPoint);
        }
        List<MatOfPoint> approxList = new ArrayList<>();
        for(int i = 0; i < hullPoints.size(); i++) {
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(new MatOfPoint2f(hullPoints.get(i).toArray()), approx, 0.01 * Imgproc.arcLength(new MatOfPoint2f(hullPoints.get(i).toArray()), true), true);
            approxList.add(new MatOfPoint(approx.toArray()));
        }
        for(MatOfPoint c: approxList) {
            Rect bBox = Imgproc.boundingRect(c);
            if(Imgproc.contourArea(c) > 10000 && c.toList().size() == 6) {
                Imgproc.drawContours(out,approxList,approxList.indexOf(c),new Scalar(255,0,0),5);
            }
        }
        showResult(canny);
        showResult(out);
        //Imgcodecs.imwrite("C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\temp2.jpg",out);
    }
    public static Mat auto_canny(Mat image) {
        MatOfDouble mean = new MatOfDouble();
        MatOfDouble std = new MatOfDouble();
        Mat edges = new Mat();
        Core.meanStdDev(image, mean, std);
        //Imgproc.adaptiveThreshold(image,image,255,Imgproc.ADAPTIVE_THRESH_MEAN_C,Imgproc.THRESH_BINARY,51,0);
        Imgproc.Canny(image, edges, mean.get(0,0)[0] - std.get(0,0)[0], mean.get(0,0)[0] + std.get(0,0)[0]);
        return edges;
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
    //Finds the MatOfPoint object that is described by a MatOfInt object
    private static MatOfPoint convertIndexesToPoints(MatOfPoint contour, MatOfInt indexes) {
        int[] arrIndex = indexes.toArray();
        Point[] arrContour = contour.toArray();
        Point[] arrPoints = new Point[arrIndex.length]; //Makes an array of points that contains the same number of elements as the MatOfInt

        for (int i=0;i<arrIndex.length;i++) {
            arrPoints[i] = arrContour[arrIndex[i]]; //Add whatever point is at the MatOfInt's value to the array of points
        }

        MatOfPoint outputHull = new MatOfPoint();
        outputHull.fromArray(arrPoints);
        return outputHull;
    }

}
