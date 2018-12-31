import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.*;

public class B {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        Mat template = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\SHAPEZ\\box.png",0);
        Mat scene = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\SHAPEZ\\box_in_scene.png",0);

        ORB orb = ORB.create();
        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        MatOfKeyPoint templateKeyPoints = new MatOfKeyPoint();
        Mat templateDescriptors = new Mat();
        Mat sceneDescriptors = new Mat();

        orb.detectAndCompute(template,new Mat(),templateKeyPoints,templateDescriptors);
        orb.detectAndCompute(scene,new Mat(),sceneKeyPoints,sceneDescriptors);

        BFMatcher bfMatcher = BFMatcher.create(BFMatcher.BRUTEFORCE_HAMMING,true);

        MatOfDMatch matches = new MatOfDMatch();

        bfMatcher.match(templateDescriptors,sceneDescriptors,matches);

        List<DMatch> matchList = matches.toList();
        List<DMatch> goodMatches = new ArrayList<>();

        Collections.sort(matchList, new Comparator<DMatch>() {
            @Override
            public int compare(DMatch o1, DMatch o2) {
                return Math.round(o1.distance-o2.distance);
            }
        });

        for(int i = 0; i < 63; i++) {
            goodMatches.add(matchList.get(i));
        }


        MatOfDMatch goodMatchesMat = new MatOfDMatch();
        goodMatchesMat.fromList(goodMatches);

        Mat display = new Mat();
        Features2d.drawMatches(template,templateKeyPoints,scene,sceneKeyPoints,goodMatchesMat,display);
        showResult(display);

        //magic?
        //[1,2,3,4,5]
        //[6,7,8,9,10]
        LinkedList<Point> objList = new LinkedList<>();
        LinkedList<Point> sceneList = new LinkedList<>();

        List<KeyPoint> keypoints_objectList = templateKeyPoints.toList();
        List<KeyPoint> keypoints_sceneList = sceneKeyPoints.toList();

        for(int i = 0; i<goodMatches.size();i++) {
            objList.addLast(keypoints_objectList.get(goodMatches.get(i).queryIdx).pt);
            sceneList.addLast(keypoints_sceneList.get(goodMatches.get(i).trainIdx).pt);
        }

        MatOfPoint2f obj = new MatOfPoint2f();
        obj.fromList(objList);

        MatOfPoint2f env = new MatOfPoint2f();
        env.fromList(sceneList);

        Mat H = Calib3d.findHomography(obj,env,Calib3d.RANSAC,5);

        Mat obj_corners = new Mat(4,1,CvType.CV_32FC2);
        Mat scene_corners = new Mat(4,1,CvType.CV_32FC2);

        obj_corners.put(0,0,new double[] {0,0});
        obj_corners.put(1,0,new double[] {template.cols(),0});
        obj_corners.put(2,0,new double[] {template.cols(),template.rows()});
        obj_corners.put(3,0,new double[] {0,template.rows()});

        Core.perspectiveTransform(obj_corners,scene_corners,H);

        Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(scene_corners,obj_corners); //

        Mat output = new Mat();
        Imgproc.warpPerspective(scene,output,perspectiveMatrix,template.size());

        showResult(output);

        Imgproc.cvtColor(scene,scene,Imgproc.COLOR_GRAY2RGB);
        Imgproc.line(scene,new Point(scene_corners.get(0,0)),new Point(scene_corners.get(1,0)),new Scalar(0,255,0),4);
        Imgproc.line(scene,new Point(scene_corners.get(1,0)),new Point(scene_corners.get(2,0)),new Scalar(0,255,0),4);
        Imgproc.line(scene,new Point(scene_corners.get(2,0)),new Point(scene_corners.get(3,0)),new Scalar(0,255,0),4);
        Imgproc.line(scene,new Point(scene_corners.get(3,0)),new Point(scene_corners.get(0,0)),new Scalar(0,255,0),4);

        showResult(scene);
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
