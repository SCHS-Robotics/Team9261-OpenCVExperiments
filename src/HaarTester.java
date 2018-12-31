import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class HaarTester {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        String cascadePath = "C:\\Users\\Cole Savage\\Desktop\\jewel2lbp.xml";

        Mat input = Imgcodecs.imread("C:\\Users\\Cole Savage\\Desktop\\20180910_095634.jpg");

        Mat gray1 = new Mat();
        Imgproc.cvtColor(input,gray1,Imgproc.COLOR_BGR2Lab);
        Mat b = new Mat();
        Core.extractChannel(gray1,b,0);

        Mat processNoiseCov = Mat.eye(4, 4, CvType.CV_32F);
        processNoiseCov = processNoiseCov.mul(processNoiseCov, 1e-4);
        System.out.println(processNoiseCov.dump());


        CascadeClassifier cascade = new CascadeClassifier(cascadePath);

        MatOfRect boundingBoxes = new MatOfRect();
        cascade.detectMultiScale(b,boundingBoxes);

        for(Rect rect : boundingBoxes.toList()) {
            Imgproc.rectangle(input, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 10);
            System.out.println("ran");
            //Imgproc.putText(input,Double.toString(levelWeights.toList().get(boundingBoxes.toList().indexOf(rect))), new Point(rect.x, rect.y), Core.FONT_HERSHEY_COMPLEX, 4, new Scalar(0, 255, 0), 5);
        }
/*
        MatOfInt rejectLevels = new MatOfInt();
        MatOfDouble levelWeightsBlue = new MatOfDouble();
        MatOfDouble levelWeightsGreen = new MatOfDouble();
        MatOfDouble levelWeightsRed= new MatOfDouble();
        MatOfDouble levelWeightsGray = new MatOfDouble();
        MatOfRect boundingBoxesBlue = new MatOfRect();
        MatOfRect boundingBoxesGreen = new MatOfRect();
        MatOfRect boundingBoxesRed = new MatOfRect();
        MatOfRect boundingBoxesGray = new MatOfRect();

        cascade.detectMultiScale3(blue,boundingBoxesBlue,rejectLevels,levelWeightsBlue,1.1,1,0,new Size(),new Size(),true);
        cascade.detectMultiScale3(green,boundingBoxesGreen,rejectLevels,levelWeightsGreen,1.1,1,0,new Size(),new Size(),true);
        cascade.detectMultiScale3(red,boundingBoxesRed,rejectLevels,levelWeightsRed,1.1,1,0,new Size(),new Size(),true);
        cascade.detectMultiScale3(gray,boundingBoxesGray,rejectLevels,levelWeightsGray,1.1,1,0,new Size(),new Size(),true);

        List<Rect> boundingBoxes = combineLists(boundingBoxesBlue.toList(),boundingBoxesGreen.toList(),boundingBoxesRed.toList(),boundingBoxesGray.toList());
        List<Double> levelWeights = combineLists(levelWeightsBlue.toList(),levelWeightsGreen.toList(),levelWeightsRed.toList(),levelWeightsGray.toList());
        for(Rect rect : boundingBoxes.toList()) {
            Imgproc.rectangle(input, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 10);
            Imgproc.putText(input,Double.toString(levelWeights.toList().get(boundingBoxes.toList().indexOf(rect))), new Point(rect.x, rect.y), Core.FONT_HERSHEY_COMPLEX, 4, new Scalar(0, 255, 0), 5);
        }

        showResult(input);

        List<Double> confidences = levelWeights;
        Collections.sort(confidences,Collections.reverseOrder());

        List<Integer> idxs = new ArrayList<>();
        idxs.add(levelWeights.indexOf(confidences.get(0)));
        idxs.add(levelWeights.indexOf(confidences.get(1)));

        List<Rect> goodRects = new ArrayList<>();
        goodRects.add(boundingBoxes.get(idxs.get(0)));
        goodRects.add(boundingBoxes.get(idxs.get(1)));

        for(Rect rect : goodRects) {
            Imgproc.rectangle(input, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 10);
            Imgproc.putText(input, "text", new Point(rect.x, rect.y), Core.FONT_HERSHEY_COMPLEX, 4, new Scalar(0, 255, 0), 5);
        }
*/
        //showResult(input);

    }

    private static <T> List<T> combineLists(List<T>... args) {
        List<T> combinedList = Stream.of(args).flatMap(i -> i.stream()).collect(Collectors.toList());   ;
        return combinedList;
    }

    private static void showResult(Mat display) {
        Mat img = display.clone();
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
}
