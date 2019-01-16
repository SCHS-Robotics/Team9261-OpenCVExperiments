import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;

public class PictographCopy {
    //Ignore this, this is the one line of code I have to use to install opencv on intellij
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {

        String inputPath = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Test Images\\left3.png";

        String templatePathLeft = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Templates\\left.jpg";
        String templatePathRight = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Templates\\right.jpg";
        String templatePathCenter = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Templates\\center.jpg";

        String templatePath =  "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Templates\\blackhexagon.jpg";

        String cascadePath = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Haar\\Test\\haarcascade_evolution_pictograph_2k3k_18st.xml";

        Mat input = Imgcodecs.imread(inputPath);

        Mat leftTemplate = Imgcodecs.imread(templatePathLeft);
        Mat rightTemplate = Imgcodecs.imread(templatePathRight);
        Mat centerTemplate = Imgcodecs.imread(templatePathCenter);

        Mat template = Imgcodecs.imread(templatePath);
        Mat result = new Mat();
        Core.MinMaxLocResult res = new Core.MinMaxLocResult();

        //Creates the Haar cascade to detect the pictograph location
        CascadeClassifier pictographCascade = new CascadeClassifier(cascadePath);

        Mat leftResult = new Mat();
        Mat rightResult = new Mat();
        Mat centerResult = new Mat();

        MatOfRect boundingBoxes = new MatOfRect();

        //Finds the bounding boxes of all the images
        pictographCascade.detectMultiScale(input,boundingBoxes);

        input = equalize(input);
        //leftTemplate = equalize(leftTemplate);
        //rightTemplate = equalize(rightTemplate);
        //centerTemplate = equalize(centerTemplate);

        Mat cropped = input.clone();

        for(Rect rect: boundingBoxes.toArray()) {
            //We only want to run this code if it only sees one pictograph
            if(boundingBoxes.toArray().length == 1) {
                Imgproc.rectangle(input, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 10);
                cropped = new Mat(input, rect); //Crops the image so that it only deals with the pixels inside the bounding box of the detected pictograph

                Imgproc.matchTemplate(cropped,template,result,Imgproc.TM_CCOEFF_NORMED);

                res = Core.minMaxLoc(result);

                Imgproc.rectangle(cropped,res.maxLoc, new Point(res.maxLoc.x + template.cols() , res.maxLoc.y + template.rows()), new Scalar(255,0,0), 2,8,0);

                //Imgproc.threshold(cropped,cropped,30,255,Imgproc.THRESH_BINARY);
            }
            else {
                System.out.println("There are multiple pictographs in this image, what are you doing....");
            }
        }
        //Puts the result on the screen, you don't need to worry about this, because we use android studio
        showResult(cropped);
        showResult(input);
    }

    //Puts the result on the screen, you don't need to worry about this, because we use android studio
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
    private static Mat equalize(Mat m) {
        ArrayList<Mat> channels = new ArrayList<>();
        int type = m.type();

        m.convertTo(m,Imgproc.COLOR_BGR2HSV);
        Core.split(m,channels);
        Imgproc.equalizeHist(channels.get(2),channels.get(2));
        Core.merge(channels,m);
        m.convertTo(m,type);
        return m;
    }
}
