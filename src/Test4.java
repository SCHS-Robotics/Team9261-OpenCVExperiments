import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Test4 {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    static CascadeClassifier hexagons = new CascadeClassifier("C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Haar\\Training Arena\\classifier\\cascade.xml");
    public static void main(String args[]) {
        Mat raw = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Test Images\\occlusion.jpg",Imgcodecs.IMREAD_GRAYSCALE);
        Mat templateImage = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Templates\\template.jpg", Imgcodecs.IMREAD_GRAYSCALE);

        ArrayList<Mat> images = new ArrayList<>();

        int i = 0;

        Mat blurred = new Mat();
        Mat warped = new Mat();
        Mat output = new Mat();

        Mat g = new Mat();

        Mat structure = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,new Size(5,5));

        List<MatOfPoint> contours = new ArrayList<>();

        Imgproc.resize(raw,raw,new Size(640,360),0,0,Imgproc.INTER_AREA);
        Imgproc.GaussianBlur(raw,blurred,new Size(7,7),0);

        Imgproc.cvtColor(raw,output,Imgproc.COLOR_GRAY2RGB);

        Mat edges = auto_canny(blurred);

        Imgproc.morphologyEx(edges,edges,Imgproc.MORPH_CLOSE,structure);

        showResult(edges);

        Imgproc.findContours(edges,contours,new Mat(),Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);

        for(MatOfPoint c : contours) {
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(new MatOfPoint2f(c.toArray()), approx, 0.01 * Imgproc.arcLength(new MatOfPoint2f(c.toArray()), true), true);
            if (approx.toList().size() == 8) {
                double area = Imgproc.contourArea(new MatOfPoint(approx.toArray()));
                if (area > 10000) {
                    warped = new Mat();
                    List<MatOfPoint> approxList = new ArrayList<>();
                    approxList.add(new MatOfPoint(approx.toArray()));
                    Imgproc.drawContours(output, approxList, 0, new Scalar(255, 0, 0), 5);

                    Mat corners = getCorners(approxList.get(0), output);
                    Mat templateCorners = getTemplateCorners(templateImage);

                    Size boxSize = getCornersSize(templateCorners);

                    Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(corners,templateCorners);
                    Imgproc.warpPerspective(raw,warped,perspectiveMatrix,boxSize);

                    Mat threshed = new Mat();
                    getHexCount(warped);
/*
                    Mat cropped = warped.submat(new Rect(0,0,(int) boxSize.width/2, (int) boxSize.height));

                    Imgproc.equalizeHist(cropped,cropped);

                    Mat b = hexFilter(cropped);

                    Imgproc.threshold(cropped,threshed,0,255,Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);

                    Mat k = new Mat();

                    Mat p = new Mat();

                    Core.bitwise_and(b,threshed,k);

                    cropped.copyTo(p,k);

                    showResult(p);
                    showResult(warped);
                    //System.out.println(getHexCount(warped, new MatOfPoint()));
*/
                }
            }
        }
        //Imgcodecs.imwrite("C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\temp.jpg",output);
        showResult(output);
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

    public static Mat getTemplateCorners(Mat template) {
        Mat output = new Mat(4,1,CvType.CV_32FC2);
        output.put(0,0, new double[] {0.0,0.0});
        output.put(1,0, new double[] {(float) template.cols(),0.0});
        output.put(2,0, new double[] {(float) template.cols(),(float) template.rows()});
        output.put(3,0, new double[] {0.0,(float) template.rows()});
        return output;
    }

    public static Size getCornersSize(Mat corners) {
        Point topLeft = new Point(corners.get(0,0));
        Point topRight = new Point(corners.get(1,0));
        Point bottomRight = new Point(corners.get(2,0));
        Point bottomLeft = new Point(corners.get(3,0));

        int width = (int) (bottomRight.x - bottomLeft.x);
        int height = (int) (bottomRight.y - topRight.y);

        return new Size(width,height);
    }

    public static Mat getCorners(MatOfPoint pts, Mat draw) {
        ArrayList<Point> ptsList = new ArrayList<>(pts.toList());
        ArrayList<Line> lines = new ArrayList<>();
        ArrayList<Point> importantPts = new ArrayList<>();
        for(int i = 0; i < ptsList.size(); i++) {

            Point init = ptsList.get(i);
            Point lookAhead = ptsList.get((i+1)%ptsList.size());

            Line temp = new Line(init, lookAhead);

            lines.add(temp);
        }
        Collections.sort(lines, new Comparator<Line>() {
            @Override
            public int compare(Line lhs, Line rhs) {
                if(lhs.length > rhs.length) {
                    return -1;
                }
                else if(lhs.length < rhs.length) {
                    return 1;
                }
                else {
                    return 0;
                }
            }
        });

        System.out.println(lines.get(0).length > lines.get(1).length);

        for(Line l : lines.subList(0,4)) {
            importantPts.add(l.start);
            importantPts.add(l.end);
        }
        importantPts.remove(0);

        Collections.sort(importantPts, new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {
                if(o1.y > o2.y){
                    return 1;
                }
                else if(o1.y < o2.y) {
                    return -1;
                }
                else {
                    return 0;
                }
            }
        });

        Line top = new Line(importantPts.get(0), importantPts.get(1));
        Line bottom = new Line(importantPts.get(importantPts.size()-2),importantPts.get(importantPts.size()-1));

        Collections.sort(importantPts, new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {
                if(o1.x > o2.x){
                    return 1;
                }
                else if(o1.x < o2.x) {
                    return -1;
                }
                else {
                    return 0;
                }
            }
        });

        Line left = new Line(importantPts.get(0),importantPts.get(1));
        Line right = new Line(importantPts.get(importantPts.size()-2),importantPts.get(importantPts.size()-1));

        Point topLeft = top.getIntersectWith(left);
        Point topRight = top.getIntersectWith(right);
        Point bottomLeft = bottom.getIntersectWith(left);
        Point bottomRight = bottom.getIntersectWith(right);

        Mat output = new Mat(4,1, CvType.CV_32FC2);

        output.put(0,0, new double[] {(float) topLeft.x,(float) topLeft.y});
        output.put(1,0, new double[] {(float) topRight.x,(float) topRight.y});
        output.put(2,0, new double[] {(float) bottomRight.x,(float) bottomRight.y});
        output.put(3,0, new double[] {(float) bottomLeft.x,(float) bottomLeft.y});

        Imgproc.line(draw,topLeft,topRight,new Scalar(0,255,0));
        Imgproc.line(draw,topLeft,bottomLeft,new Scalar(0,255,0));
        Imgproc.line(draw,bottomLeft,bottomRight,new Scalar(0,255,0));
        Imgproc.line(draw,bottomRight,topRight,new Scalar(0,255,0));

        return output;
    }

    public static Mat hexFilter(Mat input) {
        Mat output = new Mat();

        Imgproc.cvtColor(input,output,Imgproc.COLOR_GRAY2BGR);
        Imgproc.cvtColor(output,output,Imgproc.COLOR_BGR2HSV);
        Core.inRange(output,new Scalar(0,0,0), new Scalar(180,255,50), output);
        Core.bitwise_not(output,output);
        //Imgproc.GaussianBlur(output,output,new Size(5,5),0);
        //showResult(output);
        return output;
    }

    public static int getHexCount(Mat img) {

        int hexCount = 0;
        MatOfRect boxes = new MatOfRect();

        showResult(img);

        hexagons.detectMultiScale(hexFilter(img),boxes,1.1,11,0,new Size(),new Size());

        for(Rect r : boxes.toArray()) {
            Imgproc.rectangle(img,new Point(r.x,r.y),new Point(r.x+r.width,r.y+r.height),new Scalar(0,0,0),5);
        }

        showResult(img);
        return hexCount;
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
    private MatOfPoint convertIndexesToPoints(MatOfPoint contour, MatOfInt indexes) {
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
