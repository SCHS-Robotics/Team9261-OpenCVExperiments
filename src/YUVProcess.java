import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;


public class YUVProcess implements Runnable {

    private Mat input;
    private Mat output;

    public boolean running;

    public YUVProcess(Mat input,Mat output) {
        this.input = input; //input mat must be in RGBA or RGB format
        this.output = output;

        running = false;
    }

    @Override
    public void run() {
        running = true;

        Mat yuv = new Mat();
        Mat uChan = new Mat();

        Imgproc.cvtColor(input,yuv,Imgproc.COLOR_BGR2YUV);
        Core.extractChannel(yuv,uChan,1);

        yuv.release();

        Imgproc.medianBlur(uChan,uChan,9);
        Imgproc.threshold(uChan,output,145,255,Imgproc.THRESH_BINARY);

        running = false;

        uChan.release();
    }
}
