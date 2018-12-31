import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;

public class g {
    public static void main(String args[]) {
    String inputPath = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\left1.jpg";
    String templatePath = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Templates\\FIRST.jpg";

    Mat input = Imgcodecs.imread(inputPath);

    Mat left_pattern_right = Imgcodecs.imread(templatePath);

    Mat grayInput = Imgcodecs.imread(inputPath,0);
    Mat grayPattern = Imgcodecs.imread(templatePath,0);

    Mat output = new Mat();
    Mat pattern = new Mat();

    //Imgproc.equalizeHist(input, input);
    //Imgproc.equalizeHist(left_pattern_right,left_pattern_right);
/*
        Imgproc.adaptiveThreshold(grayInput,output,255,Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY,11,2);
        Imgproc.adaptiveThreshold(grayPattern,pattern,255,Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY,11,2);

        //Imgproc.matchTemplate(input,left_pattern_right,output,Imgproc.TM_CCORR_NORMED);

        //Core.MinMaxLocResult result = Core.minMaxLoc(output);

        //Imgproc.rectangle(input,result.maxLoc, new Point(result.maxLoc.x + left_pattern_right.cols() , result.maxLoc.y + left_pattern_right.rows()), new Scalar(255,0,0), 2,8,0);

        Mat structure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));

        Imgproc.morphologyEx(pattern,pattern,Imgproc.MORPH_CLOSE,structure);
        Imgproc.morphologyEx(output,output,Imgproc.MORPH_CLOSE,structure);


        Core.bitwise_not(pattern,pattern);
        Core.bitwise_not(output,output);

        ArrayList<MatOfPoint> hex = detectHexagons(pattern);

        pattern.convertTo(pattern,Imgproc.COLOR_GRAY2RGB);

        Mat e = Mat.zeros(output.size(),output.type());

        Imgproc.drawContours(e,hex,-1,new Scalar(255,255,255));
*/
        Mat res = new Mat();

        Imgproc.matchTemplate(input,left_pattern_right,res, Imgproc.TM_CCORR);

        Core.MinMaxLocResult result = Core.minMaxLoc(output);
        Imgproc.rectangle(input,result.maxLoc, new Point(result.maxLoc.x + left_pattern_right.cols() , result.maxLoc.y + left_pattern_right.rows()), new Scalar(255,0,0), 2,8,0);

        showResult(input);
    }

    private static ArrayList<MatOfPoint> detectHexagons(Mat in) {
        Mat edges = new Mat();

        MatOfDouble mean = new MatOfDouble();
        MatOfDouble std = new MatOfDouble();

        ArrayList<MatOfPoint> contours = new ArrayList<>();

        Imgproc.Canny(in, edges,0,255);
        Imgproc.findContours(edges,contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        ArrayList<MatOfPoint> hexagons = new ArrayList<>();

        /*
        MatOfInt hull = new MatOfInt();
        MatOfPoint hullPoint;
        ArrayList<MatOfPoint> hullPoints = new ArrayList<>();

        for(MatOfPoint c : contours) {
            Imgproc.convexHull(c, hull);
            hullPoint = convertIndexesToPoints(c,hull); //Finds the MatOfPoint that the hull describes
            hullPoints.add(hullPoint);
        }

        Imgproc.drawContours(edges,hullPoints,-1,new Scalar(255,255,255),5);

        Imgproc.findContours(edges,contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
*/
        for(MatOfPoint c:contours) {
            MatOfPoint2f c2f = new MatOfPoint2f();
            MatOfPoint2f approx = new MatOfPoint2f();
            c2f.fromArray(c.toArray());
            double peri = Imgproc.arcLength(c2f,true);
            Imgproc.approxPolyDP(c2f,approx,0.03*peri,true);
            if(approx.toArray().length == 6) {
                hexagons.add(new MatOfPoint(approx.toArray()));
            }
        }

        return hexagons;
    }

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
