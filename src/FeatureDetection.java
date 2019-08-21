import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.FlannBasedMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.List;

import static org.opencv.core.CvType.CV_32F;

public class FeatureDetection {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String[] args) {

        String bookObject = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Templates\\left.jpg";
        String bookScene = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Test Images\\left2.jpg";
        //String cascadePath = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Haar\\Test\\haarcascade_evolution_pictograph_2k3k_18st.xml";

        //CascadeClassifier pictographCascade = new CascadeClassifier(cascadePath);

        Mat objectImage = Imgcodecs.imread(bookObject, Imgcodecs.IMREAD_COLOR);
        Mat sceneImage = Imgcodecs.imread(bookScene, Imgcodecs.IMREAD_COLOR);

        MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
        MatOfKeyPoint objectDescriptors = new MatOfKeyPoint();

        ORB featureDetector = ORB.create();

        MatOfRect boundingBoxes = new MatOfRect();

        //Finds the bounding boxes of all the images
        //pictographCascade.detectMultiScale(sceneImage,boundingBoxes);

        for(Rect rect: boundingBoxes.toArray()) {
            //We only want to run this code if it only sees one pictograph
            if (boundingBoxes.toArray().length == 1) {
                //Imgproc.rectangle(sceneImage, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 10);
                sceneImage = new Mat(sceneImage, rect);
            }
        }

        featureDetector.detectAndCompute(objectImage, new Mat(), objectKeyPoints, objectDescriptors);

        KeyPoint[] keypoints = objectKeyPoints.toArray();

        // Create the matrix for output image.
        Mat outputImage = new Mat(objectImage.rows(), objectImage.cols(), Imgcodecs.IMREAD_COLOR);
        Scalar newKeypointColor = new Scalar(255, 0, 0);

        //Features2d.drawKeypoints(objectImage, objectKeyPoints, outputImage, newKeypointColor, 0);

        // Match object image with the scene image
        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        MatOfKeyPoint sceneDescriptors = new MatOfKeyPoint();

        featureDetector.detectAndCompute(sceneImage, new Mat(), sceneKeyPoints, sceneDescriptors);

        Mat matchoutput = new Mat(sceneImage.rows() * 2, sceneImage.cols() * 2, Imgcodecs.IMREAD_COLOR);
        Scalar matchestColor = new Scalar(0, 255, 0);

        if(sceneDescriptors.type()!=CV_32F) {
            sceneDescriptors.convertTo(sceneDescriptors, CV_32F);
            objectDescriptors.convertTo(objectDescriptors, CV_32F);
        }

        List<MatOfDMatch> matches = new LinkedList<>();
        FlannBasedMatcher descriptorMatcher = FlannBasedMatcher.create();

        System.out.println(objectDescriptors.type());
        System.out.println(sceneDescriptors.type());

        descriptorMatcher.knnMatch(objectDescriptors, sceneDescriptors, matches, 2);

        System.out.println("Calculating good match list...");
        LinkedList<DMatch> goodMatchesList = new LinkedList<>();

        float nndrRatio = 0.7f;

        for (int i = 0; i < matches.size(); i++) {
            MatOfDMatch matofDMatch = matches.get(i);
            DMatch[] dmatcharray = matofDMatch.toArray();
            DMatch m1 = dmatcharray[0];
            DMatch m2 = dmatcharray[1];

            if (m1.distance <= m2.distance * nndrRatio) {
                goodMatchesList.addLast(m1);

            }
        }

        if (goodMatchesList.size() >= 7) {
            System.out.println("Object Found!!!");

            List<KeyPoint> objKeypointlist = objectKeyPoints.toList();
            List<KeyPoint> scnKeypointlist = sceneKeyPoints.toList();

            LinkedList<Point> objectPoints = new LinkedList<>();
            LinkedList<Point> scenePoints = new LinkedList<>();

            for (int i = 0; i < goodMatchesList.size(); i++) {
                objectPoints.addLast(objKeypointlist.get(goodMatchesList.get(i).queryIdx).pt);
                scenePoints.addLast(scnKeypointlist.get(goodMatchesList.get(i).trainIdx).pt);
            }

            MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
            objMatOfPoint2f.fromList(objectPoints);
            MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
            scnMatOfPoint2f.fromList(scenePoints);

            Mat homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f, Calib3d.RANSAC, 3);

            Mat obj_corners = new Mat(4, 1, CvType.CV_32FC2);
            Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);

            obj_corners.put(0, 0, new double[]{0, 0});
            obj_corners.put(1, 0, new double[]{objectImage.cols(), 0});
            obj_corners.put(2, 0, new double[]{objectImage.cols(), objectImage.rows()});
            obj_corners.put(3, 0, new double[]{0, objectImage.rows()});

            System.out.println("Transforming object corners to scene corners...");
            Core.perspectiveTransform(obj_corners, scene_corners, homography);

            Mat img = Imgcodecs.imread(bookScene, Imgcodecs.IMREAD_COLOR);

            Imgproc.line(img, new Point(scene_corners.get(0, 0)), new Point(scene_corners.get(1, 0)), new Scalar(0, 255, 0), 4);
            Imgproc.line(img, new Point(scene_corners.get(1, 0)), new Point(scene_corners.get(2, 0)), new Scalar(0, 255, 0), 4);
            Imgproc.line(img, new Point(scene_corners.get(2, 0)), new Point(scene_corners.get(3, 0)), new Scalar(0, 255, 0), 4);
            Imgproc.line(img, new Point(scene_corners.get(3, 0)), new Point(scene_corners.get(0, 0)), new Scalar(0, 255, 0), 4);

            System.out.println("Drawing matches image...");
            MatOfDMatch goodMatches = new MatOfDMatch();
            goodMatches.fromList(goodMatchesList);

            //Features2d.drawMatches(objectImage, objectKeyPoints, sceneImage, sceneKeyPoints, goodMatches, matchoutput, matchestColor, newKeypointColor, new MatOfByte(), 2);

            showResult(outputImage);
            showResult(matchoutput);
            showResult(img);
        } else {
            System.out.println("Object Not Found");
        }
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
