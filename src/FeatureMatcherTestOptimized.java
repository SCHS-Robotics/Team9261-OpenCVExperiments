import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.BRISK;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.FlannBasedMatcher;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class FeatureMatcherTestOptimized {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}

    static String templateName = "C:\\Users\\Cole Savage\\Desktop\\Data\\2.jpg";
    static Mat template = Imgcodecs.imread(templateName,Imgcodecs.IMREAD_GRAYSCALE);

    static BRISK fd = BRISK.create();
    static Mat templateDesc = new Mat();
    static MatOfKeyPoint templatePoints = new MatOfKeyPoint();

    public static void main(String args[]) {

        String sceneName = "C:\\Users\\Cole Savage\\Desktop\\Data\\unnamed.jpg";
        Mat scene = Imgcodecs.imread(sceneName,Imgcodecs.IMREAD_GRAYSCALE);
        /*Mat scene = new Mat();

        VideoCapture cap;
        int width = 3264, height = 2448;
        cap = new VideoCapture(0);
        if (!cap.isOpened()) {
            System.out.println("Camera Error");
        } else {
            System.out.println("Camera OK?");
            cap.set(Videoio.CV_CAP_PROP_FRAME_WIDTH, width);
            cap.set(Videoio.CV_CAP_PROP_FRAME_HEIGHT, height);
        }
        try {
            Thread.sleep(1000);
        } catch (InterruptedException ex) {
        }

        cap.read(scene);
        showResult(scene);
        Imgproc.cvtColor(scene,scene,Imgproc.COLOR_BGR2GRAY);

        //Imgproc.equalizeHist(scene,scene);
*/
        //Imgproc.resize(template,template,new Size((int) Math.round(template.size().width/2),(int) Math.round(template.size().height/2)));
        //Imgproc.resize(scene,scene,new Size((int) Math.round(scene.size().width/5),(int) Math.round(scene.size().height/5)));

        long time0 = System.currentTimeMillis();

        // fd.setFastThreshold(0);
        MatOfKeyPoint scenePoints = new MatOfKeyPoint();

        Mat sceneDesc = new Mat();


        fd.detectAndCompute(scene,new Mat(),scenePoints,sceneDesc);
        fd.detectAndCompute(template,new Mat(),templatePoints,templateDesc);

        FlannBasedMatcher flann = FlannBasedMatcher.create();

        List<MatOfDMatch> matches = new ArrayList<>();

        templateDesc.convertTo(templateDesc,CvType.CV_32F);
        sceneDesc.convertTo(sceneDesc,CvType.CV_32F);

        flann.knnMatch(templateDesc,sceneDesc,matches,2);

        float ratioThresh = 0.8f;
        List<DMatch> listOfGoodMatches = new ArrayList<>();
        for (int i = 0; i < matches.size(); i++) {
            if (matches.get(i).rows() > 1) {
                DMatch[] matchList = matches.get(i).toArray();
                if (matchList[0].distance < ratioThresh * matchList[1].distance) {
                    listOfGoodMatches.add(matchList[0]);
                }
            }
        }
        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(listOfGoodMatches);

        Mat kp = new Mat();
        Features2d.drawMatches(template,templatePoints,scene,scenePoints,goodMatches,kp,new Scalar(255,0,0), new Scalar(0,0,255), new MatOfByte(), 2);
        //showResult(kp);

        LinkedList<Point> objList = new LinkedList<>();
        LinkedList<Point> sceneList = new LinkedList<>();

        List<KeyPoint> keypoints_objectList = templatePoints.toList();
        List<KeyPoint> keypoints_sceneList = scenePoints.toList();

        for(int i = 0; i < listOfGoodMatches.size(); i++){
            objList.addLast(keypoints_objectList.get(listOfGoodMatches.get(i).queryIdx).pt);
            sceneList.addLast(keypoints_sceneList.get(listOfGoodMatches.get(i).trainIdx).pt); }

        MatOfPoint2f obj = new MatOfPoint2f();
        obj.fromList(objList);

        MatOfPoint2f sc = new MatOfPoint2f();
        sc.fromList(sceneList);

        Mat H = Calib3d.findHomography(obj, sc, Calib3d.RANSAC, 6);

        Mat obj_corners = new Mat(4,1,CvType.CV_32FC2);
        Mat scene_corners = new Mat(4,1,CvType.CV_32FC2);

        obj_corners.put(0, 0, new double[] {0,0});
        obj_corners.put(1, 0, new double[] {template.cols(),0});
        obj_corners.put(2, 0, new double[] {template.cols(),template.rows()});
        obj_corners.put(3, 0, new double[] {0,template.rows()});

        Core.perspectiveTransform(obj_corners, scene_corners, H);

        Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(scene_corners,obj_corners);

        Mat output = new Mat();
        Imgproc.warpPerspective(scene,output,perspectiveMatrix,template.size());
        //showResult(output);

        Imgproc.cvtColor(scene,scene,Imgproc.COLOR_GRAY2BGR);

        Imgproc.line(scene, new Point(scene_corners.get(0,0)), new Point(scene_corners.get(1,0)), new Scalar(0, 255, 0),4);
        Imgproc.line(scene, new Point(scene_corners.get(1,0)), new Point(scene_corners.get(2,0)), new Scalar(0, 255, 0),4);
        Imgproc.line(scene, new Point(scene_corners.get(2,0)), new Point(scene_corners.get(3,0)), new Scalar(0, 255, 0),4);
        Imgproc.line(scene, new Point(scene_corners.get(3,0)), new Point(scene_corners.get(0,0)), new Scalar(0, 255, 0),4);

        //showResult(scene);

        System.out.println("time: "+Double.toString(Math.floor((System.currentTimeMillis()-time0)*100)/100.0)+"ms");


        //showResult(scene);
        //cap.release();
    }
    private static List<MatOfByte> genMask(int size) {
        List<MatOfByte> mask = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            mask.add(new MatOfByte(new byte[] {0,0}));
        }
        return mask;
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
