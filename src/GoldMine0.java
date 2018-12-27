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

public class GoldMine0 {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main (String args[]) {
        String filename = "C:\\Users\\Cole Savage\\Desktop\\Data\\40108068_320820165353263_2733329782681540191_n.jpg";
        filename = "C:\\Users\\Cole Savage\\Desktop\\Data\\20180910_095634.jpg";
        filename = "C:\\Users\\Cole Savage\\Desktop\\Data\\20180910_094912.jpg";
        //filename = "C:\\Users\\Cole Savage\\Desktop\\Data\\b.jpg";

        Mat testImg = Imgcodecs.imread(filename);

        Mat lab = new Mat();
        Mat labThresh = new Mat();
        Mat hsv = new Mat();
        Mat questionableHsv = new Mat(); //questionable
        Mat questionableThresh = new Mat();
        Mat yellowMask = new Mat();
        Mat distanceTransform = new Mat();

        Imgproc.cvtColor(testImg,lab,Imgproc.COLOR_BGR2Lab);
        Mat bChan = new Mat();
        Core.extractChannel(lab,bChan,2);
        Imgproc.threshold(bChan,labThresh,147,255,Imgproc.THRESH_BINARY);

        Imgproc.dilate(labThresh,labThresh,Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(5,5)));

        showResult(labThresh);

        Imgproc.cvtColor(testImg,hsv,Imgproc.COLOR_BGR2HSV);
        //Imgproc.cvtColor(hsv, questionableHsv, Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsv,new Scalar(0,100,100),new Scalar(23,255,255),questionableThresh);

        showResult(questionableThresh);

        Core.bitwise_and(labThresh,questionableThresh,yellowMask);

        Imgproc.distanceTransform(yellowMask, distanceTransform, Imgproc.DIST_L2, 3);
        distanceTransform.convertTo(distanceTransform, -1);

        double[] stdMean = calcStdDevMean(distanceTransform);
        double std = stdMean[0];
        double mean = stdMean[1];

        Mat test = new Mat();

        Imgproc.threshold(distanceTransform,test,mean+2*std,255,Imgproc.THRESH_BINARY);
        //Imgproc.dilate(test,test,Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(5,5)));

        test.convertTo(test,CvType.CV_8UC1);

        double med = getMedian(test);

        Mat edges = new Mat();
        double sigma = 0.33;
        Imgproc.Canny(test,edges,(int) Math.round(Math.max(0,(1-sigma)*med)),(int) Math.round(Math.min(255,1+sigma)*med));

        Imgproc.dilate(edges,edges,Imgproc.getStructuringElement(Imgproc.MORPH_CROSS,new Size(5,5)));


        List<MatOfPoint> contours = new ArrayList<>();

        Imgproc.findContours(edges,contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        int num = 0;

        for(MatOfPoint c: contours) {
            MatOfPoint2f approx = new MatOfPoint2f();
            RotatedRect bbox = Imgproc.minAreaRect(new MatOfPoint2f(c.toArray()));
            Point[] pt = new Point[4];
            bbox.points(pt);
            double a = dist(pt[0],pt[1]);
            double b = dist(pt[1],pt[2]);
            double idealArea = a*b;
            double idealPeri = 2*a+2*b;
            double realArea = Imgproc.contourArea(c);
            double realPeri = Imgproc.arcLength(new MatOfPoint2f(c.toArray()),true);
            System.out.println(Math.abs((realArea-idealArea)/idealArea));
            if(Imgproc.contourArea(c) > 5000 && Math.abs((realArea-idealArea)/idealArea) <= 1 && Math.abs((realPeri-idealPeri)/idealPeri) < 1) {
                num++;
                Imgproc.drawContours(testImg,contours,contours.indexOf(c),new Scalar(0,255,0),5);
                /*RotatedRect a = Imgproc.minAreaRect(new MatOfPoint2f(c.toArray()));
                Point[] pt = new Point[4];
                a.points(pt);
                Imgproc.line(testImg,pt[0],pt[1],new Scalar(0,0,255),20);
                Imgproc.line(testImg,pt[1],pt[2],new Scalar(0,0,255),20);
                Imgproc.line(testImg,pt[2],pt[3],new Scalar(0,0,255),20);
                Imgproc.line(testImg,pt[3],pt[0],new Scalar(0,0,255),20);*/
            }
        }



        System.out.println(num);

        //getAllMaxes(mean+7*std,new Size(50,50),distanceTransform);
        //showResult(distanceTransform);
        //showResult(edges);
        showResult(testImg);

        //Mat test = new Mat();

        //Imgproc.threshold(distanceTransform,test,mean+9*std,255,Imgproc.THRESH_BINARY);

        //showResult(testImg);
        //showResult(test);

    }

    public static RotatedRect tryMergeBoxes(RotatedRect a, RotatedRect b) {
        if(true) { //if rects can be merged

        }
        return new RotatedRect();
    }

    public static double dist(Point a, Point b) {
        return Math.sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
    }

    public static double[] calcStdDevMean(Mat input) {
        assert input.channels() == 1: "input must only have 1 channel";
        MatOfDouble std = new MatOfDouble();
        MatOfDouble mean = new MatOfDouble();
        Core.meanStdDev(input,mean,std);
        return new double[] {std.get(0,0)[0],mean.get(0,0)[0]};
    }

    public static double getMedian(Mat input) {
        Mat rowMat = input.reshape(0,1);
        Mat sorted = new Mat();
        Core.sort(rowMat,sorted,Core.SORT_ASCENDING);

        return sorted.size().width % 2 == 1 ? sorted.get(0,(int) Math.floor(sorted.size().width/2))[0] : (sorted.get(0,(int) (sorted.size().width/2)-1)[0]+sorted.get(0,(int) sorted.size().width/2)[0])/2;
    }

    public static List<Point> getAllMaxes(double minIntensity, Size minSize, Mat dstTransform) {
        List<Point> maxLocs = new ArrayList<>();
        Core.MinMaxLocResult minMax = Core.minMaxLoc(dstTransform);
        while(minMax.maxVal > minIntensity) {
            maxLocs.add(minMax.maxLoc);
            Imgproc.rectangle(dstTransform,new Point(minMax.maxLoc.x-minSize.width,minMax.maxLoc.y-minSize.height),new Point(minMax.maxLoc.x+minSize.width,minMax.maxLoc.y+minSize.height),new Scalar(0,0,0),-1);
            minMax = Core.minMaxLoc(dstTransform);
        }
        return  maxLocs;
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
