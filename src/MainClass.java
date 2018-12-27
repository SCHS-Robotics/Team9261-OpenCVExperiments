import org.opencv.core.*;

import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.LineSegmentDetector;
import org.opencv.objdetect.CascadeClassifier;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MainClass {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {

        //get pictograph detected, divide bounding box in two, crop to left half
        //for every detection, use contour matching to filter for valid detections
        //if 3 valid, center, if 2 valid, right, if 5 valid, left

        Mat template = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\Pictograph\\Templates\\edges.jpg",Imgcodecs.IMREAD_GRAYSCALE);
        Mat test = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\Pictograph\\Tests\\IMG_20170909_101838.jpg",Imgcodecs.IMREAD_GRAYSCALE);
        Mat templatelines = new Mat();
        Mat testlines = new Mat();
        Mat templateedges = Mat.zeros(template.size(),template.type());
        Mat testedges = Mat.zeros(test.size(),test.type());
        MatOfRect boxes = new MatOfRect();
        MatOfRect frame = new MatOfRect();

        List<MatOfPoint> hullPoints = new ArrayList<>();
        List<MatOfPoint> tempcontours = new ArrayList<>();
        List<MatOfPoint> testcontours;
        MatOfPoint templatecontour = new MatOfPoint();
        MatOfPoint hullPoint = new MatOfPoint();

        MatOfInt hull = new MatOfInt();

        CascadeClassifier hexagonCascade = new CascadeClassifier("C:\\TrainingArena\\trained_classifiers\\hexagoncascade2.xml");
        CascadeClassifier pictographCascade = new CascadeClassifier("C:\\Users\\Cole Savage\\Desktop\\Pictograph\\haarcascade_evolution_pictograph_2k3k_18st.xml");

        //Imgproc.equalizeHist(test,test); //to be replaced with retina

        Mat output = new Mat();
        Mat e = new Mat();

        showResult(test);
/*
        CLAHE clahe = Imgproc.createCLAHE();
        clahe.apply(test,e);
        showResult(e);
        Imgproc.GaussianBlur(e,e,new Size(7,7),1);
        test = e;
*/
        pictographCascade.detectMultiScale(test,frame);

        if(frame.toArray().length == 0) {
            System.out.println("No pictographs detected");
        }
        else {

            for (Rect r : frame.toArray()) {
                if (frame.toArray().length < 2) {
                    Rect cropRect = new Rect(new Point(r.x, r.y), new Point(Math.round(r.x + (r.width / 2)), r.y + r.height));
                    test = new Mat(test, cropRect);
                    showResult(test);
                } else {
                    System.out.println("There are two pictographs in this image, you broke the algorithm");
                }
            }

            //Imgproc.Laplacian(template,output,CvType.CV_64F);


            CLAHE clahe = Imgproc.createCLAHE();
            clahe.apply(test,e);
            showResult(e);
            Imgproc.medianBlur(e,e,17);
            test = e;

            showResult(test);

            output = template.clone();

            output.convertTo(output, CvType.CV_8UC1);

            Imgproc.Canny(output, output, 255, 255);

            //showResult(output);

            Imgproc.findContours(output, tempcontours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            templatecontour = tempcontours.get(0);

            Imgproc.drawContours(output, tempcontours, 0, new Scalar(255, 255, 255));

            //showResult(output);

            Imgproc.threshold(test, test, 68, 255, Imgproc.THRESH_BINARY);

            hexagonCascade.detectMultiScale(test, boxes);

            Core.bitwise_not(test, test);

            //Imgproc.morphologyEx(test,test,Imgproc.MORPH_CLOSE,Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,new Size(11,11)));

            List<Rect> rects = new ArrayList<>();

            int i = 1;
            int hexagonCount = 0;
            for (Rect box : boxes.toArray()) {
                testcontours = new ArrayList<>();
                System.out.println("box number " + Integer.toString(i));
                Mat cropped = new Mat(test, box);
                output = new Mat();
                Imgproc.Laplacian(cropped, output, CvType.CV_64F);

                output = cropped.clone();
                //output.convertTo(output,CvType.CV_8UC1);

                //Imgproc.Canny(output,output,20,255);
                Imgproc.findContours(output, testcontours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                for (MatOfPoint c : testcontours) {
                    double matchVal = Imgproc.matchShapes(templatecontour, c, Imgproc.CV_CONTOURS_MATCH_I1, 0);
                    System.out.println(matchVal);
                    System.out.println(c.toArray().length);
                    double peri = Imgproc.arcLength(new MatOfPoint2f(c.toArray()), true);
                    MatOfPoint2f approx = new MatOfPoint2f();
                    Imgproc.approxPolyDP(new MatOfPoint2f(c.toArray()), approx, 0.04 * peri, true);
                    System.out.println("approx" + Integer.toString(approx.toArray().length));
                    if (matchVal < 0.1 && c.toArray().length > 35 && (approx.toArray().length == 5 || approx.toArray().length == 6)) { //the 35 constant is to remove partial detections, which sometimes count as another hexagon
                    /*
                    Imgproc.convexHull(c, hull);
                    hullPoint = convertIndexesToPoints(c,hull); //Finds the MatOfPoint that the hull describes
                    Point prevPoint = null;
                    List<Double> distances = new ArrayList<>();
                    for(Point p:hullPoint.toArray()) {
                        if(prevPoint == null) {
                            prevPoint = p;
                        }
                        else {
                            distances.add(Math.sqrt(Math.pow(p.x-prevPoint.x,2)+Math.pow(p.y-prevPoint.y,2)));
                            prevPoint = p;
                        }
                        System.out.println("Sides: " + Integer.toString(distances.size()));
                    }*/
                        System.out.println("found hexagon");
                        hexagonCount++;
                        //Imgproc.drawContours(output,testcontours,testcontours.indexOf(c),new Scalar(255,255,255));
                        showResult(output);
                    }
                }

                rects.add(box);
                i++;
                //Imgproc.drawContours(output,testcontours,-1,new Scalar(255,255,255));
                //showResult(output);
            }
            System.out.println("Number of Hexagons: " + Integer.toString(hexagonCount));

            if (hexagonCount == 5) {
                System.out.println("The key is left");
            } else if (hexagonCount == 3) {
                System.out.println("The key is center");
            } else if (hexagonCount == 2) {
                System.out.println("The key is right");
            } else {
                System.out.println("Hmm");
            }

            for (Rect rect : rects) {
                Imgproc.rectangle(test, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 255, 255));
            }

            showResult(test);
        /*
        showResult(test);

        LineSegmentDetector lsd = Imgproc.createLineSegmentDetector();

        Mat prec = new Mat();
        Mat nfa = new Mat();
        Mat width = new Mat();

        lsd.detect(template,templatelines);
        lsd.detect(test,testlines);

        for (int x = 0; x < testlines.rows(); x++) {
            double[] l = testlines.get(x, 0);
            Imgproc.line(testedges, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(255, 255, 255), 1);
        }
        for (int x = 0; x < templatelines.rows(); x++) {
            double[] l = templatelines.get(x, 0);
            Imgproc.line(templateedges, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(255, 255, 255), 1);
        }
        //showResult(templateedges);

        templateedges.convertTo(templateedges, CvType.CV_8UC1);
        //testedges.convertTo(testedges, CvType.CV_8UC1);
*/
/*
        MatOfKeyPoint kptemp = new MatOfKeyPoint();
        MatOfKeyPoint kptest = new MatOfKeyPoint();

        Mat tempdesc = new Mat();
        Mat testdesc = new Mat();

        List<MatOfDMatch> matches = new ArrayList<>();
        MatOfDMatch z = new MatOfDMatch();

        FastFeatureDetector fastFeatureDetector = FastFeatureDetector.create();

        BRISK brisk = BRISK.create();

        fastFeatureDetector.detect(test,kptest);
        fastFeatureDetector.detect(template,kptemp);

        brisk.compute(test,kptest,testdesc);
        brisk.compute(template,kptemp,tempdesc);

        FlannBasedMatcher flannBasedMatcher = FlannBasedMatcher.create();
        //flannBasedMatcher.match(tempdesc,testdesc,z);

        tempdesc.convertTo(tempdesc,CvType.CV_32F);
        testdesc.convertTo(testdesc,CvType.CV_32F);

        flannBasedMatcher.knnMatch(tempdesc,testdesc,matches,2);

        Features2d.drawMatchesKnn(template,kptemp,test,kptest,matches,template);

        showResult(template);

        /*
        Imgproc.Canny(test,testedges,40,50);
        Imgproc.Canny(template,templateedges,40,50);

        showResult(testedges);
        showResult(templateedges);

        List<MatOfPoint> templatecontours = new ArrayList<>();
        List<MatOfPoint> testcontours = new ArrayList<>();
        MatOfPoint templateContour = new MatOfPoint();

        Imgproc.findContours(templateedges,templatecontours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
        templateContour = templatecontours.get(0);

        Imgproc.findContours(testedges,testcontours,new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        for(MatOfPoint c:testcontours) {
            //double g = Imgproc.matchShapes(templateContour,c,Imgproc.CV_CONTOURS_MATCH_I3,0);
            //System.out.println(g);
        }
        Mat output = new Mat(testedges.size(),test.type());
        //Imgproc.drawContours(output,testcontours,-1,new Scalar(255,255,255));
        //showResult(output);*/
        }
    }

    public static void showResult(Mat img) {
        Mat imgc = img.clone();
        Imgproc.resize(imgc, imgc, new Size(640, 480));
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", imgc, matOfByte);
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