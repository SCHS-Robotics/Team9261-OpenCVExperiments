import org.opencv.core.*;
import org.opencv.features2d.BOWKMeansTrainer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.objdetect.CascadeClassifier;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

public class HSVTest {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        Mat input = Imgcodecs.imread("C:\\subject15.wink.gif",Imgcodecs.IMREAD_GRAYSCALE);
        String cascadePath = "C:\\haarcascade_frontalface_alt.xml";
        showResult(input);
        CascadeClassifier cascade = new CascadeClassifier(cascadePath);
        MatOfRect boxes = new MatOfRect();
        cascade.detectMultiScale(input,boxes);
        for(Rect r : boxes.toArray()) {
            Imgproc.rectangle(input, new Point(r.x, r.y), new Point(r.x + r.width, r.y + r.height), new Scalar(0, 0, 0), 5);
        }

        showResult(input);
    }
    public static void showResult(Mat img) {
        Imgproc.resize(img, img, new Size(640, 480));
        //new Size(640, 480)
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
