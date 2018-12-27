import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.utils.Converters;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class test {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    private static double SZ = 20;
    public static void main(String args[]) {
        SVM svm = SVM.load("C:\\Users\\Cole Savage\\Desktop\\svm\\svm_data.dat");
        Mat image = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\svm\\training\\30_0.jpg",Imgcodecs.IMREAD_GRAYSCALE);
        showResult(image);

        if(Core.countNonZero(image)/(image.size().width*image.size().height) > 0.8) {
            Core.bitwise_not(image,image); //makes the dark pencil lines white and the paper background black
            System.out.println("ran");
        }
        Mat deskewed = deskew(image);
//new HOGDescriptor(winsize blocksize blockstride cellsize);
        HOGDescriptor hog = new HOGDescriptor(new Size(16,16),new Size(16,16), new Size(8,8), new Size(8,8),16);
        MatOfFloat descriptors = new MatOfFloat();
        hog.compute(deskewed,descriptors);

        System.out.println(descriptors.dump());

        Mat a = descriptors.reshape(1,1);
        MatOfFloat data = new MatOfFloat();
        a.convertTo(data,CvType.CV_32F);
        //Core.rotate(data,data,Core.ROTATE_180);
        System.out.println(data.cols());
        //Mat testDataMat = new Mat(1,64,CvType.CV_32FC1, Converters.Mat);
        Mat res = new Mat();
        svm.predict(data,res,0);
        System.out.println(res.dump());
        /*Imgproc.resize(image,image,new Size(20,20),0,0,Imgproc.INTER_AREA);
        Mat deskewed = deskew(image);
        HOGDescriptor hog = new HOGDescriptor(new Size(20,20),new Size(10,10), new Size(5,5), new Size(10,10),16,1,-1,0,0.2,true,64,true);
        MatOfFloat descriptors = new MatOfFloat();

        hog.compute(deskewed,descriptors);
        System.out.println(descriptors.size());
        //showResult(deskewed);
        //System.out.println(svm.predict(descriptors.reshape(1,1)));

        Mat sv = svm.getSupportVectors();
        int sv_total = sv.rows();
        Mat alpha = new Mat();
        Mat svidx = new Mat();
        double rho = svm.getDecisionFunction(0,alpha,svidx);
        Mat detector = sv.reshape(sv.rows()*sv.cols(),1);
        Mat rhoMat = new Mat(1,1,detector.type());
        rhoMat.put(0,0,new float[] {(float) rho});
        detector.push_back(rhoMat);
        //System.out.println(detector.size());
        System.out.println(hog.getDescriptorSize());
        hog.setSVMDetector(detector);*/
    }
    private static Mat deskew(Mat img) {
        Mat deskewed = new Mat();

        Moments m = Imgproc.moments(img);
        if(Math.abs(m.m02) < 1e-2) {
            return img.clone();
        }
        double skew = (float) m.mu11/m.mu02;
        Mat M = Mat.zeros(2,3, CvType.CV_32F);
        M.put(0,0, new float[] {1});
        M.put(0,1, new float[] {(float) skew});
        M.put(0,2, new float[] {(float) (-0.5*img.size().width*skew)});
        M.put(1,0, new float[] {0});
        M.put(1,1, new float[] {1});
        M.put(1,2, new float[] {0});
        System.out.println(M.dump());
        System.out.println(skew);
        Imgproc.warpAffine(img,deskewed,M,img.size(),Imgproc.WARP_INVERSE_MAP | Imgproc.INTER_LINEAR);
        return deskewed;
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
