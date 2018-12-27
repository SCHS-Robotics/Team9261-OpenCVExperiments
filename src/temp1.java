import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.video.Video;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class temp1 {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        //Mat inputtest = Imgcodecs.imread("C:\\TrainingArena\\positive_images\\jewels\\20171128_151023.jpg", Imgcodecs.IMREAD_COLOR);
        String Imagename = "IMG_5201";
        Mat inputtest = Imgcodecs.imread("C:\\TrainingArena\\positive_images\\Jewels\\"+Imagename+".jpg",Imgcodecs.IMREAD_GRAYSCALE);
        //CascadeClassifier jewel = new CascadeClassifier("C:\\TrainingArena\\classifier\\cascade.xml");
        CascadeClassifier jewel = new CascadeClassifier("C:\\TrainingArena\\trained_classifiers\\jewelcascade3.xml");
        MatOfRect boxes = new MatOfRect();
        Imgproc.GaussianBlur(inputtest,inputtest,new Size(9,9),0);
        //Imgproc.medianBlur(inputtest,inputtest,5);
        jewel.detectMultiScale(inputtest,boxes,1.1,25,0,new Size(100,100),new Size());
        MatOfInt w = new MatOfInt();
        //Objdetect.groupRectangles(boxes,w,1,1);
        //jewel.detectMultiScale(inputtest,boxes);
        //ArrayList<Rect> prevRects = new ArrayList<>();

        List<Rect> rects = boxes.toList();

        Mat draw = new Mat();
        Imgproc.cvtColor(inputtest,draw,Imgproc.COLOR_GRAY2RGB);

        for(Rect r: rects) {
            Imgproc.rectangle(draw,new Point(r.x,r.y),new Point(r.x+r.width,r.y+r.height),new Scalar(0,0,255),5);
        }
        showResult(draw);
    }

    public static Point getRectCenter(Rect rect) {
        Line a = new Line(new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height));
        Line b = new Line(new Point(rect.x+rect.width,rect.y), new Point(rect.x,rect.y+rect.height));
        Point center = a.getIntersectWith(b);
        return center;
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
