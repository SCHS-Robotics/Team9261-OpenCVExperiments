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

public class SilverMine {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        String filename = "C:\\Users\\Cole Savage\\Desktop\\Data\\40108068_320820165353263_2733329782681540191_n.jpg";
        filename = "C:\\Users\\Cole Savage\\Desktop\\Mineral_Photos\\a\\20180910_095630.jpg";
        //filename = "C:\\Users\\Cole Savage\\Desktop\\20180910_094912.jpg";
        //filename = "C:\\Users\\Cole Savage\\Desktop\\Data\\b.jpg";

        Mat inputFrame = Imgcodecs.imread(filename); //Reads in image from file, only used for testing purposes
        Imgproc.resize(inputFrame,inputFrame,new Size(inputFrame.size().width/4,inputFrame.size().height/4)); //Reduces image size for speed

        Mat hls = new Mat();
        Imgproc.cvtColor(inputFrame,hls,Imgproc.COLOR_BGR2HLS);

        Mat lum = new Mat();
        Mat thresh = new Mat();
        Core.extractChannel(hls,lum,1);
        Imgproc.threshold(lum,thresh,151,255,Imgproc.THRESH_BINARY);

        List<Mat> rgba = new ArrayList<>();
        Core.split(inputFrame,rgba);

        Mat mask = new Mat();
        Core.bitwise_and(rgba.get(0),rgba.get(1),mask);
        Core.bitwise_and(mask,rgba.get(2),mask);

        Imgproc.threshold(mask,mask,126,255,Imgproc.THRESH_BINARY);

        Core.bitwise_and(mask,thresh,mask);

        Mat k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(5,5));
        Imgproc.morphologyEx(mask,mask,Imgproc.MORPH_CLOSE,k);

        showResult(mask);

        //Calculates the median value of the image
        double med = getMedian(mask);

        //Dynamically calculates the best parameters for the Canny edge detector to find the edges of all of the detected shapes
        //Edges are represented as a binary image, with "on" pixels along the edge and "off" pixels everywhere else
        Mat edges = new Mat();
        double sigma = 0.33;
        Imgproc.Canny(mask,edges,(int) Math.round(Math.max(0,(1-sigma)*med)),(int) Math.round(Math.min(255,1+sigma)*med));

        //Enhances edge information
        Imgproc.dilate(edges,edges,Imgproc.getStructuringElement(Imgproc.MORPH_CROSS,new Size(5,5)),new Point(),1);

        showResult(edges);

        //Turns edges into a list of shapes
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(edges,contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        for(MatOfPoint c : contours) {

            Rect bbox = Imgproc.boundingRect(c);
            Point coords = new Point();
            float[] radius = new float[1];
            Imgproc.minEnclosingCircle(new MatOfPoint2f(c.toArray()),coords,radius);
            double bboxRatio = (1.0*bbox.width)/(1.0*bbox.height);
            double areaRatio = Imgproc.contourArea(c)/(Math.PI*Math.pow(radius[0],2));
            double periRatio = Imgproc.arcLength(new MatOfPoint2f(c.toArray()),true)/(2*Math.PI*radius[0]);

            Imgproc.circle(inputFrame,coords,(int) Math.floor(radius[0]),new Scalar(0,0,255));
            if(Math.abs(1-areaRatio) < 0.65 && Math.abs(1-periRatio) < 0.2 && Imgproc.contourArea(c) > 1000) {
                Imgproc.drawContours(inputFrame, contours, contours.indexOf(c), new Scalar(0, 255, 0), 9);
            }

            System.out.println(bboxRatio);
            System.out.println(areaRatio);
            System.out.println(periRatio);
            System.out.println("");
            //Imgproc.drawContours(inputFrame, contours, i, new Scalar(0, 255, 0), 9);
        }
        
        showResult(inputFrame);

        //Empties the cosmic garbage can
        System.gc();
    }
    //Prints result to the screen, only used for testing purposes
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

    //Check if polygon contains any point in a list of points. True if polygon contains any point in the list, false otherwise
    private static boolean containsPoint(MatOfPoint2f polygon, List<Point> points) {
        boolean inPolygon = false;
        for(Point p : points) {
            if(Imgproc.pointPolygonTest(polygon, p, false) >= 0) { //returns value > 0 if in polygon, 0 if on the edge of the polygon, and < 0 if outside the polygon
                inPolygon = true;
            }
        }
        return inPolygon;
    }

    //Calculates rectangle completely inside the shape
    private static Rect calcBox(MatOfPoint c) {
        //Calculates center of the shape
        Moments m = Imgproc.moments(c);
        Point center = new Point(m.m10/m.m00,m.m01/m.m00);

        //Gets list of the shape's corners
        List<Point> corners = c.toList();

        //Finds the smallest distance from the center to one of the corners
        double minDst = Integer.MAX_VALUE;
        double dst;
        for(Point p : corners) {
            dst = dist(center,p);
            minDst = dst < minDst ? dist(center,p) : minDst; //min distance equals the distance between the center and the current corner if the distance is less than the previous min distance
        }

        //Calculate and return rectangle coordinates assuming the rectangle is inside a circle of radius minDst
        return new Rect(new Point(Math.round(center.x-minDst/Math.sqrt(2)),(int) Math.round(center.y-minDst/Math.sqrt(2))),new Point(Math.round(center.x+minDst/Math.sqrt(2)),(int) Math.round(center.y+minDst/Math.sqrt(2))));
    }

    //Calculates standard deviation and mean of an image. Output is a constant list of doubles with the following format: {standard deviation, mean}
    private static double[] calcStdDevMean(Mat input) {
        assert input.channels() == 1: "input must only have 1 channel"; //Makes sure image is only 1 channel (ex: black and white)

        //Calculates image mean and standard deviation
        MatOfDouble std = new MatOfDouble();
        MatOfDouble mean = new MatOfDouble();
        Core.meanStdDev(input,mean,std);
        double[] output = new double[] {std.get(0,0)[0],mean.get(0,0)[0]};

        //Removes used images from memory to avoid overflow crashes
        std.release();
        mean.release();

        //returns output data with the following format: {standard deviation, mean}
        return output;
    }

    //Gets the median value of the image
    private static double getMedian(Mat input) {
        //Turns image into a single row of pixels
        Mat rowMat = input.reshape(0,1);

        //Sort pixel values from least to greatest
        Mat sorted = new Mat();
        Core.sort(rowMat,sorted,Core.SORT_ASCENDING);

        //Calculates median of the image. Median is the middle value of the row of sorted pixels. If there are two middle pixels, the median is their average.
        double median = sorted.size().width % 2 == 1 ? sorted.get(0,(int) Math.floor(sorted.size().width/2))[0] : (sorted.get(0,(int) (sorted.size().width/2)-1)[0]+sorted.get(0,(int) sorted.size().width/2)[0])/2;

        //Removes used images from memory to avoid overflow crashes
        rowMat.release();
        sorted.release();

        return median;
    }

    //Calculates distance between two points
    private static double dist(Point a, Point b) {
        return(Math.sqrt(Math.pow(a.x-b.x,2)+Math.pow(a.y-b.y,2))); //distance formula
    }

    //Calculates convex hull of a shape
    private static MatOfPoint hull(MatOfPoint mopIn) { //mop = MatOfPoint

        //Calculates indexes of convex points on the shape
        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(mopIn, hull, false);

        //Creates output shape to store convex points
        MatOfPoint mopOut = new MatOfPoint();
        mopOut.create((int)hull.size().height,1,CvType.CV_32SC2);

        //Selects all convex points (at the calculated hull indices) and adds them to the output shape
        for(int i = 0; i < hull.size().height ; i++)
        {
            int index = (int)hull.get(i, 0)[0];
            double[] point = new double[] {
                    mopIn.get(index, 0)[0], mopIn.get(index, 0)[1]
            };
            mopOut.put(i, 0, point);
        }

        //Removes used images from memory to avoid overflow crashes
        hull.release();

        return mopOut;
    }
}
