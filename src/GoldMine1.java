import org.opencv.bioinspired.Retina;
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

public class GoldMine1 {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main (String args[]) {

        String filename = "C:\\Users\\Cole Savage\\Desktop\\Data\\40108068_320820165353263_2733329782681540191_n.jpg";
        filename = "C:\\Users\\coles\\Desktop\\Data\\20180910_095634.jpg";
        filename = "C:\\Users\\coles\\Desktop\\Data\\20180910_094912.jpg";
        //filename = "C:\\Users\\coles\\Desktop\\Data\\b.jpg";
        filename = "C:\\Users\\coles\\Desktop\\Data\\failure\\IMG-1734.jpg";
        Mat input = Imgcodecs.imread(filename); //Reads in image from file, only used for testing purposes
        Imgproc.resize(input, input, new Size(320, (int) Math.round((320/input.size().width)*input.size().height))); //Reduces image size for speed

        Retina retina = Retina.create(input.size());
        retina.setup("C:\\Users\\coles\\Desktop\\Data\\RetinaParams2.xml");
        retina.clearBuffers();

        retina.applyFastToneMapping(input,input);

        Mat yuv = new Mat();
        Imgproc.cvtColor(input,yuv,Imgproc.COLOR_RGB2YUV);
        Mat uChan = new Mat();
        Core.extractChannel(yuv,uChan,1);
        Mat b = new Mat();
        Imgproc.medianBlur(uChan,uChan,9);
        //Imgproc.filter2D(uChan,uChan,-1,Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(1,1)));
        //showResult(b);
        //showResult(uChan);
        Mat a = new Mat();
        Imgproc.threshold(uChan,a,145,255,Imgproc.THRESH_BINARY);
        showResult(a);

        Imgproc.cvtColor(input,input,Imgproc.COLOR_BGR2RGBA); //Converts input image from BGR to RGBA, only used for testing purposes

        //Defines all Mats that will be used in the program
        Mat lab = new Mat();
        Mat labThreshBinary = new Mat();
        Mat labThreshOtsu = new Mat();
        Mat labThresh = new Mat();
        Mat hsv = new Mat();
        Mat hChan = new Mat();
        Mat bChan = new Mat();

        //Converts input from RGB color format to Lab color format, then extracts the b channel
        //Lab is based on the opponent color model, and the b channel represents the blue-yellow axis, so it will be useful in finding yellow colors
        Imgproc.cvtColor(input,lab,Imgproc.COLOR_RGB2Lab);
        Core.extractChannel(lab,bChan,2);

        //Removes used images from memory to avoid overflow crashes
        lab.release();

        /*Thresholds the b channel in two different ways to get a binary filter (correct or not correct)
        for all detected yellow pixels
        The binary threshold selects all pixels with a b value above 145
        The Otsu threshold does the same thing as the binary threshold, but tries to dynamically
        select the threshold value (the value above which a pixel is considered yellow) to divide the
        image by contrast. The binary threshold is very inclusive, for reasons that will become clear later*/

        double stdm1[] = calcStdDevMean(bChan);

        Imgproc.threshold(bChan,labThreshBinary,stdm1[1],255,Imgproc.THRESH_BINARY);
        Imgproc.threshold(bChan,labThreshOtsu,0,255,Imgproc.THRESH_OTSU);

        //showResult(labThreshBinary);

        /*Otsu threshold will usually do a good job of segmenting the cubes from the rest of the
        image (as they contrast heavily with the background), but does not function well when there
        are no cubes in the image, as the optimal contrast threshold will not necessarily be filtering
        for yellow. The binary threshold, however, is not affected by the absence of the cubes, and so
        by performing a bitwise and of the images, we keep only the area where both thresholds agree,
        which accounts for times when the cube is not in the image while keeping the otsu threshold's power*/
        Core.bitwise_and(labThreshBinary,labThreshOtsu,labThresh);

        //showResult(labThreshBinary);
        //showResult(labThreshOtsu);

        //Removes used images from memory to avoid overflow crashes
        bChan.release();
        labThreshBinary.release();
        labThreshOtsu.release();

        //Converts input from RGB color format to HSV color format, then extracts the h channel
        //HSV stands for hue, saturation, value. We are only interested in the h channel, which stores color information
        //Because of its division of color into a separate channel, HSV format is resistant to lighting changes and so is good for color filtering
        Imgproc.cvtColor(input,hsv,Imgproc.COLOR_RGB2HSV_FULL);
        Core.extractChannel(hsv,hChan,0);

        Mat sChan = new Mat();
        Core.extractChannel(hsv,sChan,1);

        Mat temp = new Mat(hChan.size(),hChan.type(),new Scalar(50.59));

        Mat temp2 = new Mat();
        Core.absdiff(hChan,temp,temp2);

        Core.bitwise_not(temp2,temp2);

        Core.bitwise_and(temp2,sChan,temp2);

        //Imgproc.medianBlur(temp2,temp2,9);

        showResult(temp2);

        double stdm2[] = calcStdDevMean(temp2);

        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(temp2);

        System.out.println(getMedian(temp2));
        System.out.println(stdm2[1]);
        System.out.println(stdm2[0]);

        double mul = (minMaxLocResult.maxVal-stdm2[1])/stdm2[0];

        System.out.println();

        Imgproc.threshold(temp2,temp2,0.75*minMaxLocResult.maxVal,255,Imgproc.THRESH_BINARY);

        //showResult(temp2);

        //showResult(sChan);

        //showResult(labThresh);

        Core.bitwise_and(temp2,labThresh,labThresh);

        //showResult(temp2);

        //showResult(labThresh);

        Core.bitwise_and(labThresh,a,labThresh);

        //showResult(labThresh);

        //showResult(temp2);

        Imgproc.morphologyEx(labThresh,labThresh,Imgproc.MORPH_CLOSE,Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(5,5)));

        showResult(labThresh);

        //Masks image so that the only h regions detected are those that were also detected by the Lab otsu and binary thresholds
        Mat masked = new Mat();
        Core.bitwise_and(hChan,labThresh,masked);

        //Removes used images from memory to avoid overflow crashes
        hsv.release();
        hChan.release();

        /*Computes the distance transform of the Lab image threshold and then does a binary threshold of that
        The distance transform sorts pixels by their distance from the nearest black pixel. Larger distance means a higher value
        This is done here in order to reduce noise, which will have a small distance from black pixels*/
        Mat distanceTransform = new Mat();
        Mat thresholded = new Mat();
        Imgproc.distanceTransform(labThresh,distanceTransform,Imgproc.DIST_L2,3);
        distanceTransform.convertTo(distanceTransform,CvType.CV_8UC1);


        Mat msk = new Mat();
        Imgproc.threshold(distanceTransform,msk,0,255,Imgproc.THRESH_BINARY);

        double stdm[] = calcStdDevMean(distanceTransform,msk);

        Imgproc.threshold(distanceTransform,thresholded,stdm[1]/stdm[0],255,Imgproc.THRESH_BINARY);

        //showResult(thresholded);

        //showResult(thresholded);
        //Removes used images from memory to avoid overflow crashes
        distanceTransform.release();

        //Performs a gaussian blur on the threshold to help eliminate remaining high frequency noise
        Imgproc.GaussianBlur(thresholded,thresholded,new Size(3,3),0);

        //Finds all detected blobs in the thresholded distance transform and finds their centers
        List<MatOfPoint> centerShapes = new ArrayList<>();
        Imgproc.findContours(thresholded,centerShapes,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
        thresholded.release();
        List<Point> centers = new ArrayList<>();
        for(MatOfPoint shape : centerShapes) {
            Moments moments = Imgproc.moments(shape);
            centers.add(new Point(moments.m10/moments.m00,moments.m01/moments.m00));
            shape.release();
        }

        //Removes used images from memory to avoid overflow crashes
        //labThresh.release();

        //Imgproc.bilateralFilter(masked,dst,5,5,5);
        //Imgproc.GaussianBlur(labThresh,labThresh,new Size(5,5),0);

        //Calculates the median value of the image
        double med = getMedianNonZero(masked);

        //Dynamically calculates the best parameters for the Canny edge detector to find the edges of all of the detected shapes
        //Edges are represented as a binary image, with "on" pixels along the edge and "off" pixels everywhere else
        Mat edges = new Mat();
        double sigma = 0.33;
        Imgproc.Canny(masked,edges,(int) Math.round(Math.max(0,(1-sigma)*med)),(int) Math.round(Math.min(255,1+sigma)*med));

        showResult(edges);

        //showResult(masked);

        //Enhances edge information
        Imgproc.dilate(edges,edges,Imgproc.getStructuringElement(Imgproc.MORPH_CROSS,new Size(2,2)),new Point(),1);

        //showResult(edges);

        //Turns edges into a list of shapes
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(edges,contours,new Mat(),Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);

        //showResult(edges);

        //Removes used images from memory to avoid overflow crashes
        edges.release();

        int detected = 0;
        List<Double> usedx = new ArrayList<>();
        List<Double> usedy = new ArrayList<>();
        //Loops through the list of shapes (contours) and finds the ones most likely to be a cube
        for (int i = 0; i < contours.size(); i++) {
            //Approximates the shape to smooth out excess edges
            MatOfPoint2f approx = new MatOfPoint2f();
            double peri = Imgproc.arcLength(new MatOfPoint2f(contours.get(i).toArray()), true);
            Imgproc.approxPolyDP(new MatOfPoint2f(contours.get(i).toArray()), approx, 0.03 * peri, true); //0.1 is a detail factor, higher factor = lower detail, lower factor = higher detail
            MatOfPoint approxMop = new MatOfPoint(approx.toArray());

            //Does a simple size check to eliminate extremely small contours
            if (Imgproc.contourArea(approxMop) > 100) {

                //Checks if one of the distance transform centers is contained within the shape
                Rect box = calcBox(contours.get(i));
                Imgproc.putText(input,"1",new Point(box.x, box.y), Imgproc.FONT_HERSHEY_COMPLEX, 1, new Scalar(0, 0, 0), 3);
                Point center = getCenter(approxMop);
                if (containsPoint(approx,centers) && !(usedx.contains(center.x) && usedy.contains(center.y))) {

                    usedx.add(center.x);
                    usedy.add(center.y);
                    //Calculates a convex hull of the shape, covering up any dents
                    MatOfPoint convex = hull(approxMop);
                    //Calculates a rectangle that lies completely inside the shape
                    Rect bbox = calcBox(convex);

                    //Size check to see if the box could be calculated
                    if (bbox.x >= 0 && bbox.y >= 0 && bbox.x + bbox.width <= masked.cols() && bbox.y + bbox.height <= masked.rows()) {
                        //Selects the region of interest (roi) determined from the calcBox function from the masked h channel image
                        Mat roi = masked.submat(bbox);
                        //Calculates the standard deviation and mean of the selected region. In this case it will calculate the average color and the color standard deviation
                        double[] stdMean = calcStdDevMean(roi);

                        //Does a test for average color and standard deviation (average color between 10 and 40, exclusive, and standard deviation less than 24)

                        System.out.println(stdMean[1]);
                        System.out.println(stdMean[0]);
                        System.out.println();

                        if (stdMean[1] > 12 && stdMean[1] < 51 && stdMean[0] < 24) {
                            //Calculate the overall bounding rectangle around the shape
                            Rect bboxLarge = Imgproc.boundingRect(convex);

                            List<MatOfPoint> approxList = new ArrayList<>();
                            approxList.add(convex);


                            //Imgproc.drawContours(input, contours, i, new Scalar(255, 0, 0), 9);
                            //Imgproc.drawContours(input,approxList,-1,new Scalar(0,0,255),9);
                            Imgproc.putText(input, Double.toString(Math.floor(100*(1.0 * bboxLarge.width) / (1.0 * bboxLarge.height))/100.0), new Point(bbox.x, bbox.y), Imgproc.FONT_HERSHEY_COMPLEX, 1, new Scalar(255, 0, 0), 3);

                            //Checks the size of the bounding box against what it can be based on a model of a rotating cube. Tolerance is added to account for noise
                            double tolerance = 0.2; //must be positive
                            //((1.0 * bboxLarge.width) / (1.0 * bboxLarge.height)) > Math.sqrt(2.0 / 3.0) * (1 - tolerance) && ((1.0 * bboxLarge.width) / (1.0 * bboxLarge.height)) < Math.sqrt(3.0 / 2.0) * (1 + tolerance)
                            if (true) {
                                //Checks if shape has 4 or 6 corners, which will be true for any cube-shaped object

                                if (convex.toList().size() == 4 || convex.toList().size() == 5 || convex.toList().size() == 6) {
                                    //Draws shape to screen
                                    Imgproc.drawContours(input, approxList, 0, new Scalar(0, 255, 0), 9);
                                    detected++;
                                    Imgproc.putText(input, Integer.toString(detected),new Point(bbox.x, bbox.y), Imgproc.FONT_HERSHEY_COMPLEX, 3, new Scalar(255, 0, 255), 3);
                                    approxList.clear();
                                }
                            }
                        }
                    }
                }
            }
            //Removes used images from memory to avoid overflow crashes
            contours.get(i).release();
            approx.release();
            approxMop.release();
        }

        System.out.println(detected);
        //Removes used images from memory to avoid overflow crashes
        masked.release();

        //Prints result to the screen, only used for testing purposes
        Imgproc.cvtColor(input,input,Imgproc.COLOR_BGR2RGBA);
        showResult(input);

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

    private static Point getCenter(MatOfPoint c) {
        Moments m = Imgproc.moments(c);
        return new Point(m.m10/m.m00,m.m01/m.m00);
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
    private static double[] calcStdDevMean(Mat input, Mat mask) {
        assert input.channels() == 1: "input must only have 1 channel"; //Makes sure image is only 1 channel (ex: black and white)

        //Calculates image mean and standard deviation
        MatOfDouble std = new MatOfDouble();
        MatOfDouble mean = new MatOfDouble();
        Core.meanStdDev(input,mean,std,mask);
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
    private static double getMedianNonZero(Mat input) {
        //Turns image into a single row of pixels
        Mat rowMat = input.reshape(0,1);

        //Sort pixel values from least to greatest
        Mat sorted = new Mat();
        Core.sort(rowMat,sorted,Core.SORT_ASCENDING);

        double sum = 0;
        int idx = 0;
        int loops = 0;
        while(sum == 0 && loops < sorted.cols()) {
            sum+=sorted.get(0,loops)[0];
            idx+=sorted.get(0,loops)[0] > 0 ? 1 : 0;
            loops++;
        }

        //Calculates median of the image. Median is the middle value of the row of sorted pixels. If there are two middle pixels, the median is their average.
        double median = (sum != 0 ) ? ((sorted.size().width-idx) % 2 == 1 ? sorted.get(0,(int) Math.floor(idx+((sorted.size().width-idx)/2)))[0] : (sorted.get(0,(int) (idx+(sorted.size().width-idx)/2)-1)[0]+sorted.get(0,(int) (idx+(sorted.size().width-idx)/2))[0])/2) : 0;

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
