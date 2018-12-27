import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class DetectPictograph {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        Mat template = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\Pictograph\\Templates\\left.jpg",Imgcodecs.IMREAD_GRAYSCALE);
        Mat test = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\Pictograph\\Templates\\right.jpg",Imgcodecs.IMREAD_GRAYSCALE);

        MatOfKeyPoint kptemp = new MatOfKeyPoint();
        MatOfKeyPoint kptest = new MatOfKeyPoint();

        Mat tempdesc = new Mat();
        Mat testdesc = new Mat();

        LinkedList<MatOfDMatch> matches = new LinkedList<>();
        MatOfDMatch z = new MatOfDMatch();

        ORB  orb = ORB.create();

        orb.detect(test,kptest);
        orb.detect(template,kptemp);

        orb.compute(test,kptest,testdesc);
        orb.compute(template,kptemp,tempdesc);

        FlannBasedMatcher flannBasedMatcher = FlannBasedMatcher.create();
        //flannBasedMatcher.match(tempdesc,testdesc,z);

        List<DMatch> good_matches = new ArrayList<>();

        tempdesc.convertTo(tempdesc,CvType.CV_32F);
        testdesc.convertTo(testdesc, CvType.CV_32F);

        flannBasedMatcher.knnMatch(tempdesc,testdesc,matches,2);

        for (Iterator<MatOfDMatch> iterator = matches.iterator(); iterator.hasNext();) {
            MatOfDMatch matOfDMatch = iterator.next();
            if (matOfDMatch.toArray()[0].distance / matOfDMatch.toArray()[1].distance < 0.7) {
                good_matches.add(matOfDMatch.toArray()[0]);
            }
        }

        // get keypoint coordinates of good matches to find homography and remove outliers using ransac
        List<Point> pts1 = new ArrayList<>();
        List<Point> pts2 = new ArrayList<>();
        for(int i = 0; i< good_matches.size(); i++){
            pts1.add(kptemp.toList().get(good_matches.get(i).queryIdx).pt);
            pts2.add(kptest.toList().get(good_matches.get(i).trainIdx).pt);
        }

        // convertion of data types - there is maybe a more beautiful way
        Mat outputMask = new Mat();
        MatOfPoint2f pts1Mat = new MatOfPoint2f();
        pts1Mat.fromList(pts1);
        MatOfPoint2f pts2Mat = new MatOfPoint2f();
        pts2Mat.fromList(pts2);

        // Find homography - here just used to perform match filtering with RANSAC, but could be used to e.g. stitch images
        // the smaller the allowed reprojection error (here 15), the more matches are filtered
        Mat Homog = Calib3d.findHomography(pts1Mat, pts2Mat, Calib3d.RANSAC, 15, outputMask, 2000, 0.995);

        // outputMask contains zeros and ones indicating which matches are filtered
        LinkedList<DMatch> better_matches = new LinkedList<DMatch>();
        for (int i = 0; i < good_matches.size(); i++) {
            if (outputMask.get(i, 0)[0] != 0.0) {
                better_matches.add(good_matches.get(i));
            }
        }

        // DRAWING OUTPUT
        Mat outputImg = new Mat();
        // this will draw all matches, works fine
        MatOfDMatch better_matches_mat = new MatOfDMatch();
        better_matches_mat.fromList(better_matches);
        Features2d.drawMatches(template, kptemp, test, kptest, better_matches_mat, outputImg);

        LinkedList<Point> cornerList = new LinkedList<>();
        cornerList.add(new Point(0,0));
        cornerList.add(new Point(template.cols(),0));
        cornerList.add(new Point(template.cols(),template.rows()));
        cornerList.add(new Point(0,template.rows()));

        MatOfPoint obj_corners = new MatOfPoint();
        obj_corners.fromList(cornerList);

        MatOfPoint scene_corners = new MatOfPoint();

        Core.perspectiveTransform(obj_corners, scene_corners, Homog);

        // save image
        showResult(outputImg);

        //Features2d.drawMatches2(template,kptemp,test,kptest,matches,template);
    }
    public static void showResult(Mat img) {
        Mat imgc = img.clone();
        //Imgproc.resize(imgc, imgc, new Size(640, 480));
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
}
