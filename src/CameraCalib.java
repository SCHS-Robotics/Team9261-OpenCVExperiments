import org.opencv.aruco.Aruco;
import org.opencv.aruco.CharucoBoard;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;


public class CameraCalib {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String[] args) {
        //Mat chess = Imgcodecs.imread("C:\\Users\\coles\\Desktop\\temp\\chess.png",Imgcodecs.IMREAD_GRAYSCALE);

        File[] directory = new File("C:\\Users\\coles\\Desktop\\temp").listFiles();

        List<Mat> objectPoints = new ArrayList<>();
        List<Mat> imagePoints = new ArrayList<>();

        Mat distCoeffs = new Mat();

        List<Mat> rvecs = new ArrayList<>();
        List<Mat> tvecs = new ArrayList<>();

        Mat intrinsic = new Mat(3,3,CvType.CV_32FC1);
        intrinsic.put(0, 0, 1);
        intrinsic.put(1, 1, 1);

        MatOfPoint3f obj = new MatOfPoint3f();

        Mat test = Imgcodecs.imread("C:\\Users\\coles\\Desktop\\temp\\right05.jpg");

        //Imgproc.resize(test,test,new Size(320,240));

        for (int i = 0; i < 9*6; i++) {
            obj.push_back(new MatOfPoint3f(new Point3(i/9,i%6,0.0f)));
        }

        for(int i = 0; i < directory.length; i++){

            Mat image = Imgcodecs.imread(directory[i].getAbsolutePath(),Imgcodecs.IMREAD_UNCHANGED); //for each file, read the image

            //Imgproc.resize(image,image,new Size(320,240));

            MatOfPoint2f corners = new MatOfPoint2f();
            boolean found = Calib3d.findChessboardCorners(image, new Size(9,6),corners, Calib3d.CALIB_CB_ADAPTIVE_THRESH + Calib3d.CALIB_CB_NORMALIZE_IMAGE + Calib3d.CALIB_CB_FAST_CHECK);
            System.out.println(found);
            if(found) {
                TermCriteria term = new TermCriteria(TermCriteria.EPS | TermCriteria.MAX_ITER, 30, 0.1);
                Imgproc.cornerSubPix(image, corners, new Size(11, 11), new Size(-1, -1), term);
                objectPoints.add(obj);
                imagePoints.add(corners);
            }

            Calib3d.drawChessboardCorners(image,new Size(9,6),corners,found);

            showResult(image);

            image.release();
            System.gc();
        }

        double x = Calib3d.calibrateCamera(objectPoints,imagePoints,test.size(),intrinsic,distCoeffs,rvecs,tvecs);

        System.out.println(x);

        Mat undistorted = new Mat();
        Calib3d.undistort(test,undistorted,intrinsic,distCoeffs);
        showResult(test);
        showResult(undistorted);

/*
        Imgproc.resize(chess,chess,new Size(640,480));
        MatOfPoint2f corners = new MatOfPoint2f();
        boolean found = Calib3d.findChessboardCorners(chess, new Size(9,6),corners, Calib3d.CALIB_CB_ADAPTIVE_THRESH + Calib3d.CALIB_CB_NORMALIZE_IMAGE + Calib3d.CALIB_CB_FAST_CHECK);
        System.out.println(found);
        if(found) {
            TermCriteria term = new TermCriteria(TermCriteria.EPS | TermCriteria.MAX_ITER, 30, 0.1);
            Imgproc.cornerSubPix(chess, corners, new Size(11, 11), new Size(-1, -1), term);
        }
        Imgproc.cvtColor(chess,chess,Imgproc.COLOR_GRAY2BGR);

        Calib3d.drawChessboardCorners(chess,new Size(9,6),corners,found);

        showResult(chess);*/
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
