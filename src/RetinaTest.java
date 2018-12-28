import org.opencv.bioinspired.Bioinspired;
import org.opencv.bioinspired.Retina;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

public class RetinaTest {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        Mat input = Imgcodecs.imread("C:\\Users\\coles\\OneDrive\\Desktop\\Data\\darkcat.jpg");
        Retina retina = Retina.create(input.size());
        //retina.write("C:\\Users\\coles\\OneDrive\\Desktop\\Data\\DefaultRetinaParams.xml");
        retina.setup("C:\\Users\\coles\\OneDrive\\Desktop\\Data\\RetinaParams2.xml");
        System.out.println(retina.printSetup());
        Mat gamma = new Mat();
        input.convertTo(input, CvType.CV_32F);
        Core.normalize(input,input,0,255,Core.NORM_MINMAX,-1,new Mat());
        Core.pow(input,1.0/5.0,gamma);
        retina.clearBuffers();
        retina.run(gamma);
        Mat parvo = new Mat(input.size(),input.type());
        Mat magno = new Mat(input.size(),input.type());
        Mat test = new Mat();
        retina.applyFastToneMapping(input,test);
        showResult(input);
        showResult(test);
        retina.getParvo(parvo);
        retina.getMagno(magno);

        //showResult(input);
        showResult(parvo);
        //showResult(magno);
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
