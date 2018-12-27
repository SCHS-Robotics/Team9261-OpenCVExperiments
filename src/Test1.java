import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.CLAHE;
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

public class Test1 {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        Mat img_object = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\Data\\1.png",Imgcodecs.IMREAD_GRAYSCALE);
        Mat img_scene = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\Data\\IMG-1723.jpg",Imgcodecs.IMREAD_GRAYSCALE);

        if(img_scene.width() > img_object.width() && img_scene.height() > img_object.height()) {
            double ratio = img_object.size().width/img_scene.size().width;
            Imgproc.resize(img_scene,img_scene,new Size(img_object.width(),Math.round(img_scene.height()*ratio)));
        }

        ORB detector = ORB.create();

        detector.setFastThreshold(0);

        MatOfKeyPoint keypoints_object = new MatOfKeyPoint();
        MatOfKeyPoint keypoints_scene  = new MatOfKeyPoint();

        detector.detect(img_object, keypoints_object);
        detector.detect(img_scene, keypoints_scene);

        ORB extractor = ORB.create(); //SURF, someday...;

        Mat descriptor_object = new Mat();
        Mat descriptor_scene = new Mat() ;

        extractor.compute(img_object, keypoints_object, descriptor_object);
        extractor.compute(img_scene, keypoints_scene, descriptor_scene);

        BFMatcher matcher = BFMatcher.create(BFMatcher.BRUTEFORCE_HAMMING,true);
        MatOfDMatch matches = new MatOfDMatch();

        matcher.match(descriptor_object, descriptor_scene, matches);
        List<DMatch> matchesList;
        matchesList = matches.toList();

        double max_dist = 0.0;
        double min_dist = Integer.MAX_VALUE;

        System.out.println(matches.dump());

        for(Object d:matchesList.toArray()){
            DMatch m = (DMatch) d;
            double dist = m.distance;
            System.out.println(dist);
            min_dist = dist < min_dist ? dist : min_dist;
            max_dist = dist > max_dist ? dist : max_dist;
            System.out.println(dist);
        }

        System.out.println("-- Max dist : " + max_dist);
        System.out.println("-- Min dist : " + min_dist);

        LinkedList<DMatch> good_matches = new LinkedList<>();
        MatOfDMatch gm = new MatOfDMatch();

        for(DMatch ma : matchesList){
            if(ma.distance < 2.5*min_dist){
                good_matches.addLast(ma);
            }
        }

        gm.fromList(good_matches);

        Mat img_matches = new Mat();
        Features2d.drawMatches(
                img_object,
                keypoints_object,
                img_scene,
                keypoints_scene,
                gm,
                img_matches,
                new Scalar(255,0,0),
                new Scalar(0,0,255),
                new MatOfByte(),
                2);

        showResult(img_matches);

        LinkedList<Point> objList = new LinkedList<>();
        LinkedList<Point> sceneList = new LinkedList<>();

        List<KeyPoint> keypoints_objectList = keypoints_object.toList();
        List<KeyPoint> keypoints_sceneList = keypoints_scene.toList();

        for(int i = 0; i<good_matches.size(); i++){
            objList.addLast(keypoints_objectList.get(good_matches.get(i).queryIdx).pt);
            sceneList.addLast(keypoints_sceneList.get(good_matches.get(i).trainIdx).pt); }

        MatOfPoint2f obj = new MatOfPoint2f();
        obj.fromList(objList);

        MatOfPoint2f scene = new MatOfPoint2f();
        scene.fromList(sceneList);

        Mat H = Calib3d.findHomography(obj, scene, Calib3d.RANSAC,0);

        Mat obj_corners = new Mat(4,1,CvType.CV_32FC2);
        Mat scene_corners = new Mat(4,1,CvType.CV_32FC2);

        obj_corners.put(0, 0, new double[] {0,0});
        obj_corners.put(1, 0, new double[] {img_object.cols(),0});
        obj_corners.put(2, 0, new double[] {img_object.cols(),img_object.rows()});
        obj_corners.put(3, 0, new double[] {0,img_object.rows()});

        Core.perspectiveTransform(obj_corners, scene_corners, H);

        Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(scene_corners,obj_corners);

        Mat output = new Mat();
        Imgproc.warpPerspective(img_scene,output,perspectiveMatrix,img_scene.size());
        showResult(output);

        img_scene.convertTo(img_scene,Imgproc.COLOR_GRAY2BGR);

        Imgproc.line(img_scene, new Point(scene_corners.get(0,0)), new Point(scene_corners.get(1,0)), new Scalar(0, 0, 0),4);
        Imgproc.line(img_scene, new Point(scene_corners.get(1,0)), new Point(scene_corners.get(2,0)), new Scalar(0, 0, 0),4);
        Imgproc.line(img_scene, new Point(scene_corners.get(2,0)), new Point(scene_corners.get(3,0)), new Scalar(0, 0, 0),4);
        Imgproc.line(img_scene, new Point(scene_corners.get(3,0)), new Point(scene_corners.get(0,0)), new Scalar(0, 0, 0),4);

        showResult(img_scene);

        Imgcodecs.imwrite("C:\\Users\\Cole Savage\\Desktop\\Pictograph\\matches.jpg",img_matches);

        //showResult(img_matches);
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
}
