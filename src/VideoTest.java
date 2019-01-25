import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class VideoTest {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        VideoCapture cap = new VideoCapture("C:\\Users\\Cole Savage\\Desktop\\Data\\20180910_095748.mp4");
        Mat mat = new Mat();

        // Grab the first frame to get the dimensions
        cap.read(mat);

        int w = mat.cols();
        int h = mat.rows();

        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), BufferedImage.TYPE_3BYTE_BGR);
        byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();

        // Create a window to display the video
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        ImageIcon icon = new ImageIcon(image);

        JLabel label = new JLabel(icon);
        frame.getContentPane().add(label);

        // Resize it to fit the video
        //frame.pack();
        frame.setVisible(true);

       // KalmanTracker kalmanTracker = new KalmanTracker(new Point(0,0));

        List<KalmanTracker> kalmanTrackers = new ArrayList<>();

        long startTime = System.currentTimeMillis();
        for (;;) {
            Imgproc.resize(mat, mat, new Size(320, (int) Math.round((320/mat.size().width)*mat.size().height)));
            List<Rect> rects = GoldMineVideoProcess.main(mat);

            for(Rect r : rects) {
                Point center = new Point(r.x+r.width/2.0,r.y+r.height/2.0);
                if(!pointInAnyBoxes(rects,center)) {

                }
            }
/*
            if((p.x != 0 || p.y != 0)) {
                kalmanTracker.update(p,true);

            }

            Point prediction = kalmanTracker.getPrediction();

            Imgproc.circle(mat,prediction,5,new Scalar(0,0,255),-1);
*/
            showResult(mat);

            // Copy pixels from the Mat to the image
            mat.get(0, 0, data);

            // Refresh the display
            label.repaint();

            // Grab the next frame
            cap.read(mat);
        }
    }


    public static BufferedImage mat2Img(Mat in)
    {
        BufferedImage out;
        byte[] data = new byte[320 * 240 * (int)in.elemSize()];
        int type;
        in.get(0, 0, data);

        if(in.channels() == 1)
            type = BufferedImage.TYPE_BYTE_GRAY;
        else
            type = BufferedImage.TYPE_3BYTE_BGR;

        out = new BufferedImage(320, 240, type);

        out.getRaster().setDataElements(0, 0, 320, 240, data);
        return out;
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

    private static boolean pointInAnyBoxes(List<Rect> rects, Point p) {
        boolean inAnyBox = false;
        for(Rect r : rects) {
            inAnyBox = pointInBox(r,p);
        }
        return inAnyBox;
    }

    private static boolean pointInBox(Rect r, Point p) {
        return p.x >= r.x && p.y >= r.y && p.x <= r.x+r.width && p.y <=r.y+r.height;
    }
}
