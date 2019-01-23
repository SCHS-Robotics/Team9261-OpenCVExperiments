import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class HSV_LabProcess implements Runnable {

    private Mat input;
    private Mat output;
    private Mat intensityMap;

    public boolean running;

    public HSV_LabProcess(Mat input, Mat intensityMap, Mat output) {
        this.input = input;
        this.intensityMap = intensityMap;
        this.output = output;

        running = false;
    }

    @Override
    public void run() {

        running = true;

        Mat hsv = new Mat();
        Mat sChan = new Mat();

        Imgproc.cvtColor(input,hsv,Imgproc.COLOR_RGB2HSV_FULL);
        Core.extractChannel(hsv,sChan,1);

        LabProcess labProcess = new LabProcess(input,sChan,output);
        HSVProcess hsvProcess = new HSVProcess(hsv,sChan,intensityMap);

        labProcess.run();
        hsvProcess.run();

        while(labProcess.running || hsvProcess.running);

        running = false;

        hsv.release();
        sChan.release();

    }
}
