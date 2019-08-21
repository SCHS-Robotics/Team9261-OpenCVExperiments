import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.objdetect.CascadeClassifier;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.*;

public class Test2 {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    private static CascadeClassifier hexagons = new CascadeClassifier("C:\\TrainingArena2\\classifier\\cascade.xml");
    //private static int start = 370;
    private static Mat templatemoments = new Mat(new Size(1,7),CvType.CV_64F);
    public static void main(String args[]) {

        //Creates a template Mat of HuMoments for the pictograph shape
        templatemoments.put(0,0,new double[] {0.1669123645304541});
        templatemoments.put(1,0,new double[] {0.001193746112655853});
        templatemoments.put(2,0,new double[] {0.0001429867580420092});
        templatemoments.put(3,0,new double[] {4.707644290035998*Math.pow(10,-6)});
        templatemoments.put(4,0,new double[] {-1.206552593594882*Math.pow(10,-10)});
        templatemoments.put(5,0,new double[] {-1.476030688273523*Math.pow(10,-7)});
        templatemoments.put(6,0,new double[] {-1.89782952890317*Math.pow(10,-11)});

        String Imagename = "rot.jpg";

        Mat raw = Imgcodecs.imread("C:\\TrainingArena\\positive_images\\pictographs\\"+Imagename,Imgcodecs.IMREAD_GRAYSCALE);
        Mat templateImage = Imgcodecs.imread("C:\\TrainingArena\\positive_images\\pictographs\\left2.jpg", Imgcodecs.IMREAD_GRAYSCALE);


        Mat blurred = new Mat();
        Mat output = new Mat();

        Mat warped;

        //Mat OCRSeparate = new Mat();

        //Mat structure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(1,1));

        List<MatOfPoint> contours = new ArrayList<>();

        Imgproc.resize(raw,raw,new Size(640,(640/raw.size().width)*raw.size().height),0,0,Imgproc.INTER_AREA);
        Imgproc.GaussianBlur(raw,blurred,new Size(7,7),0);

        Imgproc.cvtColor(raw,output,Imgproc.COLOR_GRAY2RGB);

        Mat edges = auto_canny(blurred);

        //Imgproc.morphologyEx(edges,edges,Imgproc.MORPH_DILATE,structure);

        // showResult(edges);

        Imgcodecs.imwrite("C:\\Users\\Cole Savage\\Desktop\\Pictograph\\edges.jpg",edges);

        Imgproc.findContours(edges,contours,new Mat(),Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);

        for(MatOfPoint c : contours) {
            MatOfInt intHull = new MatOfInt();
            Imgproc.convexHull(c,intHull);
            MatOfPoint hull = convertIndexesToPoints(c,intHull);
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(new MatOfPoint2f(hull.toArray()), approx, 0.01 * Imgproc.arcLength(new MatOfPoint2f(hull.toArray()), true), true);
            if (approx.toList().size() == 6) {
                double area = Imgproc.contourArea(new MatOfPoint(approx.toArray()));
                if (area > 10000) {
                    warped = new Mat();
                    List<MatOfPoint> approxList = new ArrayList<>();
                    approxList.add(new MatOfPoint(approx.toArray()));
                    Moments moments = Imgproc.moments(approx);
                    Imgproc.drawContours(output, approxList, 0, new Scalar(0, 0, 255), 4);
                    Mat huMoments = new Mat(new Size(1,7),CvType.CV_64F);
                    Imgproc.HuMoments(moments,huMoments);
                    boolean isValid = false;
                    double avg = 0;
                    //System.out.println(huMoments.dump());
                    //System.out.println(templatemoments.dump());
                    /*for(int j = 0; j < 7; j++) {
                        //System.out.println(huMoments.get(j,0)[0]);
                        double ratio = huMoments.get(j,0)[0]/templatemoments.get(j,0)[0];
                        avg += ratio;
                        System.out.println(Math.abs(ratio));
                    }*/

                    double ratio1 = huMoments.get(0,0)[0]/templatemoments.get(0,0)[0];
                    double ratio2 = huMoments.get(1,0)[0]/templatemoments.get(1,0)[0];
                    double ratio3 = huMoments.get(2,0)[0]/templatemoments.get(2,0)[0];
                    double ratio4 = huMoments.get(3,0)[0]/templatemoments.get(3,0)[0];
                    double ratio5 = huMoments.get(4,0)[0]/templatemoments.get(4,0)[0];
                    double ratio6 = huMoments.get(5,0)[0]/templatemoments.get(5,0)[0];
                    double ratio7 = huMoments.get(6,0)[0]/templatemoments.get(6,0)[0];
                    if(Math.abs(1 - Math.abs(ratio1)) <= 0.3) {
                        avg++;
                    }
                    if(Math.abs(1 - Math.abs(ratio2)) <= 1.16) {
                        avg++;
                    }
                    if(Math.abs(1 - Math.abs(ratio3)) <= 2.2) {
                        avg++;
                    }
                    if(Math.abs(1 - Math.abs(ratio4)) <= 0.4) {
                        avg++;
                    }
                    if(Math.abs(1 - Math.abs(ratio5)) <= 100) {
                        avg++;
                    }
                    if(Math.abs(1 - Math.abs(ratio6)) <= 12) {
                        avg++;
                    }
                    if(Math.abs(1 - Math.abs(ratio7)) <= 500) {
                        avg++;
                    }

                    System.out.println(ratio1);
                    System.out.println(ratio2);
                    System.out.println(ratio3);
                    System.out.println(ratio4);
                    System.out.println(ratio5);
                    System.out.println(ratio6);
                    System.out.println(ratio7);

                    System.out.println("average: "+Double.toString(avg));
                    System.out.println("");
                    System.out.println(avg >= 4);
                    if(avg >= 4) {
                        Imgproc.drawContours(output, approxList, 0, new Scalar(255, 0, 0), 4);
                        isValid = true;
                    }
                    showResult(output);
                    if(isValid) {
                        Mat corners = getCorners(approxList.get(0), moments, output);
                        Mat templateCorners = getTemplateCorners(templateImage);

                        Size boxSize = getCornersSize(templateCorners);

                        Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(corners, templateCorners);
                        Imgproc.warpPerspective(raw, warped, perspectiveMatrix, boxSize);

                        Mat cropped = warped.submat(new Rect(0, 0, (int) boxSize.width / 2, (int) boxSize.height));

                        Mat temp = warped.clone();

                        int count = getHexCount(cropped, temp);

                        if (count == 2) {
                            System.out.println("right");
                        } else if (count == 3) {
                            System.out.println("center");
                        } else if (count == 5) {
                            System.out.println("left");
                        } else {
                            System.out.println("unknown");
                        }
                    }
                }
            }
        }
        //Imgcodecs.imwrite("C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\temp.jpg",warped);
        showResult(output);
    }

    private static Mat auto_canny(Mat image) {
        MatOfDouble mean = new MatOfDouble();
        MatOfDouble std = new MatOfDouble();
        Mat edges = new Mat();
        Core.meanStdDev(image, mean, std);
        //Imgproc.adaptiveThreshold(image,image,255,Imgproc.ADAPTIVE_THRESH_MEAN_C,Imgproc.THRESH_BINARY,51,0);
        Imgproc.Canny(image, edges, mean.get(0,0)[0] - std.get(0,0)[0], mean.get(0,0)[0] + std.get(0,0)[0]);
        return edges;
    }

    private static Mat getTemplateCorners(Mat template) {
        Mat output = new Mat(4,1,CvType.CV_32FC2);
        output.put(0,0, new double[] {0.0,0.0});
        output.put(1,0, new double[] {(float) template.cols(),0.0});
        output.put(2,0, new double[] {(float) template.cols(),(float) template.rows()});
        output.put(3,0, new double[] {0.0,(float) template.rows()});
        return output;
    }

    private static Size getCornersSize(Mat corners) {
        Point topRight = new Point(corners.get(1,0));
        Point bottomRight = new Point(corners.get(2,0));
        Point bottomLeft = new Point(corners.get(3,0));

        int width = (int) (bottomRight.x - bottomLeft.x);
        int height = (int) (bottomRight.y - topRight.y);

        return new Size(width,height);
    }

    private static Mat getCorners(MatOfPoint pts, Moments moments, Mat draw) {
        List<Point> ptsList = new ArrayList<>(pts.toList());
        List<Line> lines = new ArrayList<>();

        Line bottom;
        Line top;
        Line left;
        Line right;

        for(int i = 0; i < ptsList.size(); i++) {

            Point init = ptsList.get(i);
            Point lookAhead = ptsList.get((i+1)%ptsList.size());

            Line temp = new Line(init, lookAhead);

            lines.add(temp);
        }

        List<List<Line>> opposites = new ArrayList<>();
        List<Line> used = new ArrayList<>();
        for(Line line: lines) {
            List<Line> pair = new ArrayList<>();
            for(Line nextLine: lines) {

                double angle = Math.toDegrees(Math.atan((line.m - nextLine.m) / (1 + line.m * nextLine.m)));
                if(Double.isNaN(angle)) {
                    angle = Double.isInfinite(line.m) ? 90 - Math.abs(Math.toDegrees(Math.atan(nextLine.m))) : 90 - Math.abs(Math.toDegrees(Math.atan(line.m)));
                }
                if(Math.abs(angle) > 0 && Math.abs(angle) < 15 && !(used.contains(line) || used.contains(nextLine))) {
                    pair.add(line);
                    pair.add(nextLine);
                    opposites.add(pair);
                    used.addAll(pair);
                    //System.out.println(Integer.toString(lines.indexOf(line)) + " and " + Integer.toString(lines.indexOf(nextLine)) + ": " + Double.toString(angle));
                    //line.drawLabeled(draw,Integer.toString(lines.indexOf(line)));
                    //nextLine.drawLabeled(draw,Integer.toString(lines.indexOf(nextLine)));
                    break;
                }
                //System.out.println(Integer.toString(lines.indexOf(line)) + " and " + Integer.toString(lines.indexOf(nextLine)) + ": " + Double.toString(angle));

                pair.clear();
            }
        }

        //This is SPAGETTI CODE
        List<Integer> indices = new ArrayList<>();
        List<Integer> topBottomLoc = new ArrayList<>();
        boolean bottomFound = false;
        for(Line line1 : opposites.get(0)) {
            if(bottomFound) {
                break;
            }
            for(Line line2 : opposites.get(1)) {
                Point intersection = line1.getIntersectWith(line2);
                if(ptsList.contains(intersection)) {
                    System.out.println(Integer.toString(lines.indexOf(line1)) + " and " + Integer.toString(lines.indexOf(line2)));
                    if(indices.contains(lines.indexOf(line1))) {
                        int pairIndex = opposites.get(0).contains(line1) ? 0 : 1;
                        int index = opposites.get(pairIndex).indexOf(line1);
                        topBottomLoc.add(pairIndex);
                        topBottomLoc.add(index);
                        bottomFound = true;
                        break;
                    }
                    else if(indices.contains(lines.indexOf(line2))) {
                        int pairIndex = opposites.get(0).contains(line2) ? 0 : 1;
                        int index = opposites.get(pairIndex).indexOf(line2);
                        topBottomLoc.add(pairIndex);
                        topBottomLoc.add(index);
                        bottomFound = true;
                        break;
                    }
                    else {
                        indices.add(lines.indexOf(line1));
                        indices.add(lines.indexOf(line2));
                    }
                }
            }
        }
        bottom = opposites.get(topBottomLoc.get(0)).get(topBottomLoc.get(1));
        top = opposites.get(topBottomLoc.get(0)).get((1-topBottomLoc.get(1))%2);

        Line side1 = opposites.get((1-topBottomLoc.get(0))%2).get(0);
        Line side2 = opposites.get((1-topBottomLoc.get(0))%2).get(1);

        Point center = new Point(moments.m10/moments.m00,moments.m01/moments.m00);
        Point side1StartRotated;
        Point side2StartRotated;
        double angle = Math.atan(bottom.m);
        if((top.start.y+top.end.y)/2 >= (bottom.end.y+bottom.start.y)/2) { //because we go in a circle to create the lines
            side1StartRotated = rotate(side1.start,center,Math.PI-angle);
            side2StartRotated = rotate(side2.start,center,Math.PI-angle);
        }
        else {
            side1StartRotated = rotate(side1.start,center,-angle);
            side2StartRotated = rotate(side2.start,center,-angle);
        }

        //Imgproc.circle(draw,side1StartRotated,5,new Scalar(255,255,255),-1);
        //Imgproc.circle(draw,side1.start,5,new Scalar(255,255,255),-1);

        //Imgproc.circle(draw,top.start,5,new Scalar(255,255,255),-1);
        //Imgproc.circle(draw,bottom.end,5,new Scalar(255,255,255),-1);

        //Imgproc.circle(draw,rotate(top.start,center,Math.PI+angle),5,new Scalar(0,255,255),-1);
        //Imgproc.circle(draw,rotate(bottom.end,center,Math.PI+angle),5,new Scalar(0,255,255),-1);

        left = side1StartRotated.x < side2StartRotated.x ? side1 : side2;
        right = side1StartRotated.x < side2StartRotated.x ? side2 : side1;

        Point topLeft = top.getIntersectWith(left);
        Point topRight = top.getIntersectWith(right);
        Point bottomLeft = bottom.getIntersectWith(left);
        Point bottomRight = bottom.getIntersectWith(right);

        Imgproc.circle(draw,bottomLeft,4,new Scalar(0,0,255),-1);
        Imgproc.circle(draw,bottomRight,4,new Scalar(0,0,255),-1);
        Imgproc.circle(draw,topLeft,4,new Scalar(0,0,255),-1);
        Imgproc.circle(draw,topRight,4,new Scalar(0,0,255),-1);

        Mat output = new Mat(4,1, CvType.CV_32FC2);

        output.put(0,0, new double[] {(float) topLeft.x,(float) topLeft.y});
        output.put(1,0, new double[] {(float) topRight.x,(float) topRight.y});
        output.put(2,0, new double[] {(float) bottomRight.x,(float) bottomRight.y});
        output.put(3,0, new double[] {(float) bottomLeft.x,(float) bottomLeft.y});

        Imgproc.line(draw,topLeft,topRight,new Scalar(0,255,0));
        Imgproc.line(draw,topLeft,bottomLeft,new Scalar(0,255,0));
        Imgproc.line(draw,bottomLeft,bottomRight,new Scalar(0,255,0));
        Imgproc.line(draw,bottomRight,topRight,new Scalar(0,255,0));

        //showResult(draw);

        return output;
    }

    private static int getHexCount(Mat img, Mat draw) {

        int hexCount = 0;
        MatOfRect boxes = new MatOfRect();

        hexagons.detectMultiScale(img,boxes,1.1,3,0,new Size(),new Size());

        for(Rect r : boxes.toArray()) {
            Imgproc.rectangle(draw,new Point(r.x,r.y),new Point(r.x+r.width,r.y+r.height),new Scalar(0,0,0),5);
            hexCount++;
            //Mat cropped = img.submat(r);
            //Imgcodecs.imwrite("C:\\TrainingArena\\positive_images\\pictographs\\hexagons\\"+Integer.toString(start)+".jpg",cropped);
            //start++;
        }

        showResult(draw);
        return hexCount;
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

    private static Point rotate(Point p, Point origin, double angle) {
        double xdiff = p.x - origin.x;
        double ydiff = p.y - origin.y;
        double cos = Math.cos(angle);
        double sin = Math.sin(angle);
        double x = origin.x + cos*xdiff - sin*ydiff;
        double y = origin.y + sin*xdiff + cos*ydiff;
        return new Point(x,y);
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