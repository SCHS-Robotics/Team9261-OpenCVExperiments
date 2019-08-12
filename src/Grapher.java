import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.plot.Plot2d;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.Random;

public class Grapher {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}

    public static void main(String args[]) {

        final int maxX = 10;

        Random random = new Random();

        Mat xData = new Mat();
        Mat yData = new Mat();

        xData.convertTo(xData,CvType.CV_64F);
        yData.convertTo(xData,CvType.CV_64F);

        double t = 0;

        double lastTimeStep = 0;

        while(t <= 30) {

            if(t <= maxX) {

                Mat x = new Mat(1,1,CvType.CV_64F);
                Mat y = new Mat(1,1,CvType.CV_64F);

                x.put(0,0,t);
                y.put(0,0,9*Math.sin(t)+1);

                xData.push_back(x);
                yData.push_back(y);
            }
            else {

                //shift left then draw to empty space

                //System.out.println(xData.dump());

                double lastX = xData.get(xData.rows()-1,0)[0];

                xData = xData.submat(1,xData.rows(),0,1);
                Mat newX = Mat.ones(1,1,CvType.CV_64F);

                Core.multiply(newX,new Scalar(lastX+lastTimeStep),newX); //fix, newx should be different
                xData.push_back(newX);
                Core.subtract(xData,new Scalar(lastTimeStep),xData);

                double xShift = xData.get(0,0)[0];

                Core.subtract(xData,new Scalar(xShift),xData);

                System.out.println(t-10*Math.floor(t/10.0));

                yData = yData.submat(1,yData.rows(),0,1);
                yData.push_back(Mat.zeros(1,1,CvType.CV_64F));
                yData.put(yData.rows()-1,0,9*Math.sin(t)+1);

                //System.out.println(xData.dump());
                //System.out.println();
                //System.out.println(yData.dump());
            }

            //System.out.println(t);
            //System.out.println();
            //System.out.println();

            Plot2d plotter = Plot2d.create(xData,yData);
            plotter.setInvertOrientation(true);
            plotter.setMaxX(maxX);
            plotter.setMaxY(10);
            plotter.setMinX(0);
            plotter.setMinY(-10);
            //plotter.setPlotSize(1280,720);
            plotter.setShowText(false);

            Mat plot = new Mat();
            plotter.render(plot);
            showResult(plot);

            System.out.println(xData.reshape(1,1).dump());

            lastTimeStep = 0.1*(random.nextInt(4)+1);
            t+=lastTimeStep;
                    //0.1*(random.nextInt(2)+1);
        }
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
