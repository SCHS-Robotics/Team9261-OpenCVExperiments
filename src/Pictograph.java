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
import java.util.Collections;

public class Pictograph {
    //Ignore this, this is the one line of code I have to use to install opencv on intellij
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {

        String inputPath = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Test Images\\right2.jpg";

        String templatePathLeft = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Templates\\left.jpg";
        String templatePathRight = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Templates\\right.jpg";
        String templatePathCenter = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Main\\Pictographs\\Templates\\center.jpg";

        String cascadePath = "C:\\Users\\Cole Savage\\Desktop\\OpenCV Images\\Haar\\Test\\haarcascade_evolution_pictograph_2k3k_18st.xml";

        Mat input = Imgcodecs.imread(inputPath);

        Mat leftTemplate = Imgcodecs.imread(templatePathLeft);
        Mat rightTemplate = Imgcodecs.imread(templatePathRight);
        Mat centerTemplate = Imgcodecs.imread(templatePathCenter);

        //Creates the Haar cascade to detect the pictograph location
        CascadeClassifier pictographCascade = new CascadeClassifier(cascadePath);

        Mat leftResult = new Mat();
        Mat rightResult = new Mat();
        Mat centerResult = new Mat();

        MatOfRect boundingBoxes = new MatOfRect();

        //Finds the bounding boxes of all the images
        pictographCascade.detectMultiScale(input,boundingBoxes);

        input = equalize(input);
        leftTemplate = equalize(leftTemplate);
        rightTemplate = equalize(rightTemplate);
        centerTemplate = equalize(centerTemplate);

        Mat cropped = input.clone();

        for(Rect rect: boundingBoxes.toArray()) {
            //We only want to run this code if it only sees one pictograph
            if(boundingBoxes.toArray().length == 1) {
                Imgproc.rectangle(input, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 10);
                cropped = new Mat(input, rect); //Crops the image so that it only deals with the pixels inside the bounding box of the detected pictograph

                //Imgproc.threshold(cropped,cropped,30,255,Imgproc.THRESH_BINARY);

                /*
               if(cropped.size().height != centerTemplate.size().height) {

                   double heightScale = centerTemplate.size().height/cropped.size().height;

                   Size newSize = new Size(cropped.size().width * heightScale, centerTemplate.height());

                   System.out.println(leftTemplate.size().height);
                   System.out.println(cropped.size().height);

                   System.out.println(heightScale);

                   Imgproc.resize(cropped,cropped,newSize);
                }
                */
                //Tries to match each template with the cropped image

                Imgproc.matchTemplate(cropped,leftTemplate,leftResult,Imgproc.TM_CCOEFF_NORMED);
                Imgproc.matchTemplate(cropped,rightTemplate,rightResult,Imgproc.TM_CCOEFF_NORMED);
                Imgproc.matchTemplate(cropped,centerTemplate,centerResult,Imgproc.TM_CCOEFF_NORMED);

                //Gets the location where the templates match up the best or the worst
                Core.MinMaxLocResult minMaxResultLeft = Core.minMaxLoc(leftResult);
                Core.MinMaxLocResult minMaxResultRight = Core.minMaxLoc(rightResult);
                Core.MinMaxLocResult minMaxResultCenter = Core.minMaxLoc(centerResult);

                ArrayList<Double> results = new ArrayList<>();

                //maxVal will be a measure of how well the template matches the image
                results.add(minMaxResultLeft.maxVal);
                results.add(minMaxResultRight.maxVal);
                results.add(minMaxResultCenter.maxVal);

                //Gets the index of the best match
                double max = Collections.max(results);
                int index = results.indexOf(max);

                String key = "";
                Core.MinMaxLocResult result;

                //Switch case actually doesn't look that bad here
                switch(index) {
                    case 0:
                        key = "left";
                        result = Core.minMaxLoc(leftResult);
                        break;
                    case 1:
                        key = "right";
                        result = Core.minMaxLoc(rightResult);
                        break;
                    case 2:
                        key = "center";
                        result = Core.minMaxLoc(centerResult);
                        break;
                    default:
                        key = "Something is really wrong, you should probably fix whatever it is";
                        result = Core.minMaxLoc(leftResult);
                        break;
                }

                System.out.println(key);
                //Draw rectangle around the matched image
                Imgproc.rectangle(cropped,result.maxLoc, new Point(result.maxLoc.x + leftTemplate.cols() , result.maxLoc.y + leftTemplate.rows()), new Scalar(255,0,0), 2,8,0);

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
