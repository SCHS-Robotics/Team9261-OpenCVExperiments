import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class HSVProcess implements Runnable {

    private Mat hsv;
    private Mat sChan;
    private Mat output;

    public boolean running;

    public HSVProcess(Mat input, Mat sChan, Mat output) {
        this.hsv = input; //input should be HSV_FULL format
        this.sChan = sChan;
        this.output = output;

        running = false;
    }

    @Override
    public void run() {

        running = true;

        Mat hChan = new Mat();
        Mat diff = new Mat();
        Mat sChanNorm = new Mat();

        Core.extractChannel(hsv,hChan,0);
        Mat targetColor = new Mat(hChan.size(),hChan.type(),new Scalar(50.59));

        Core.absdiff(hChan,targetColor,diff);
        Core.bitwise_not(diff,diff);

        hChan.release();
        targetColor.release();

        Core.normalize(sChan,sChanNorm,0,255,Core.NORM_MINMAX);
        Core.bitwise_and(diff,sChanNorm,diff);

        sChanNorm.release();

        Imgproc.medianBlur(diff,output,9);

        running = false;

        diff.release();
    }
}
