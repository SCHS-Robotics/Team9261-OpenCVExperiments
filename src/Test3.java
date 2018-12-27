import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Test3 {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    static CascadeClassifier hexagons = new CascadeClassifier("C:\\TrainingArena\\trained_classifiers\\hexagoncascade2.xml");
    public static void main(String args[]) {
        Mat raw = Imgcodecs.imread("C:\\TrainingArena\\positive_images\\pictographs\\center.jpg",Imgcodecs.IMREAD_GRAYSCALE);
        Mat templateImage = Imgcodecs.imread("C:\\TrainingArena\\positive_images\\pictographs\\left2.jpg", Imgcodecs.IMREAD_GRAYSCALE);

        //showResult(raw);

        ArrayList<Mat> images = new ArrayList<>();

        int i = 0;

        Mat blurred = new Mat();
        Mat warped = new Mat();
        Mat output = new Mat();

        Mat g = new Mat();

        Mat structure = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,new Size(7,7));

        List<MatOfPoint> contours = new ArrayList<>();

        Imgproc.resize(raw,raw,new Size(640,(640/raw.size().width)*raw.size().height),0,0,Imgproc.INTER_AREA);
        Imgproc.GaussianBlur(raw,blurred,new Size(7,7),0);

        Imgproc.cvtColor(raw,output,Imgproc.COLOR_GRAY2RGB);

        Mat edges = auto_canny(blurred);

        //Imgproc.morphologyEx(edges,edges,Imgproc.MORPH_CLOSE,structure);

        showResult(edges);

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
                    List<MatOfPoint> approxList = new ArrayList<>();
                    approxList.add(new MatOfPoint(approx.toArray()));
                    Imgproc.drawContours(output, approxList, 0, new Scalar(0, 0, 255), 5);
                    Moments moments = Imgproc.moments(approx);

                    Mat templatemoments = new Mat(new Size(1,7),CvType.CV_64F);
                    templatemoments.put(0,0,new double[] {0.1669123645304541});
                    templatemoments.put(1,0,new double[] {0.001193746112655853});
                    templatemoments.put(2,0,new double[] {0.0001429867580420092});
                    templatemoments.put(3,0,new double[] {4.707644290035998*Math.pow(10,-6)});
                    templatemoments.put(4,0,new double[] {-1.206552593594882*Math.pow(10,-10)});
                    templatemoments.put(5,0,new double[] {-1.476030688273523*Math.pow(10,-7)});
                    templatemoments.put(6,0,new double[] {-1.89782952890317*Math.pow(10,-11)});
                    Mat huMoments = new Mat(new Size(1,7),CvType.CV_64F);
                    Imgproc.HuMoments(moments,huMoments);

                    //System.out.println(templatemoments.size());
                    //System.out.println(huMoments.size());
                    //System.out.println(templatemoments.dump());

                    for(int j = 0; j < 7; j++) {
                        double ratio = huMoments.get(i,0)[0]/templatemoments.get(i,0)[0];
                        System.out.println(Math.abs(ratio));
                        if(Math.abs(1-ratio) < 0.2) {
                            System.out.println(Math.abs(1-ratio));
                            Imgproc.drawContours(output, approxList, 0, new Scalar(255, 0, 0), 4);
                        }
                    }

                    MatOfDouble xor = new MatOfDouble();
                    //System.out.println(huMoments.dump());
                }
            }
        }
        //Imgcodecs.imwrite("C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\temp.jpg",warped);
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
        List<Line> success = new ArrayList<>();
        List<Line> failure = new ArrayList<>();
        for(Line line: lines) {
            Line nextLine = lines.get((lines.indexOf(line)+1)%lines.size());

            double angle = (180/Math.PI)*Math.atan((line.m - nextLine.m) / (1 + line.m * nextLine.m));

            if(Math.abs(90 - Math.abs(angle)) <= 25 && !success.contains(line)) {
                success.add(line);
                line.draw(draw);
            }
            if(Math.abs(90 - Math.abs(angle)) <= 25 && !success.contains(nextLine)) {
                success.add(nextLine);
                nextLine.draw(draw);
            }
            if(!(Math.abs(90 - Math.abs(angle)) <= 25)) {
                failure.add(line);
            }
            System.out.println(angle);
        }

        showResult(draw);

        Collections.sort(success, new Comparator<Line>() {
            @Override
            public int compare(Line o1, Line o2) {
                return (int) Math.round(o2.m-o1.m);
            }
        });

        /*
        combine two lists
        sort list by slope
        compare successive elements to see if they are equal
         */

        double maxDistance = 0;
        double secondDistance = 0;

        int bottomIndex = 0;

        List<Integer> indices = new ArrayList<>();

        indices.add(0);
        indices.add(0);
        indices.add(0);
        indices.add(0);

        for(Line line:success) {
            Line nextLine = success.get((success.indexOf(line)+1)%success.size());
            if(Math.abs(line.m-nextLine.m) > maxDistance) {
                secondDistance = maxDistance;
                indices.set(2,indices.get(0));
                indices.set(3,indices.get(1));
                maxDistance = Math.abs(line.m-nextLine.m);
                indices.set(0,success.indexOf(line));
                indices.set(1,success.indexOf(nextLine));
            }
            else if(Math.abs(line.m-nextLine.m) > secondDistance) {
                secondDistance = Math.abs(line.m-nextLine.m);
                indices.set(2,success.indexOf(line));
                indices.set(3,success.indexOf(nextLine));
            }
        }

        Collections.sort(indices);

        for(int index:indices) {
            List sub = indices.subList(index+1,4);
            System.out.println(sub);
            if(sub.contains(index)) {
                bottomIndex = index;
                break;
            }
        }

        bottom = success.get(bottomIndex);

        success.remove(bottom);

        Collections.sort(failure, new Comparator<Line>() {
            @Override
            public int compare(Line o1, Line o2) {
/*
                if(Math.abs(bot.m-o1.m) > Math.abs(bot.m-o2.m)) {
                    return 1;
                }
                else {
                    return -1;
                }
                */
                return (int) Math.round(Math.abs(bottom.m-o1.m)-Math.abs(bottom.m-o2.m));
            }
        });

        top = failure.get(0);

        Collections.sort(success, new Comparator<Line>() {
            @Override
            public int compare(Line o1, Line o2) {
                return (int) Math.round(o2.b-o1.b);
            }
        });
        showResult(draw);
        if(top.b < bottom.b) {
            left = success.get(0);
            right = success.get(1);
        }
        else {
            left = success.get(1);
            right = success.get(0);
            System.out.println("Upsidown");
        }

        bottom.draw(draw);

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

        return output;
    }

    public static int getHexCount(Mat img, Mat draw) {

        int hexCount = 0;
        MatOfRect boxes = new MatOfRect();

        hexagons.detectMultiScale(img,boxes,1.1,3,0,new Size(),new Size());

        for(Rect r : boxes.toArray()) {
            Imgproc.rectangle(draw,new Point(r.x,r.y),new Point(r.x+r.width,r.y+r.height),new Scalar(0,0,0),5);
            hexCount++;
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