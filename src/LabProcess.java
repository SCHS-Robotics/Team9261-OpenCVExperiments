import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.imgproc.Imgproc;

public class LabProcess implements Runnable {

    private Mat input;
    private Mat sChan;
    private Mat output;

    public boolean running;

    public LabProcess(Mat input, Mat sChan, Mat output) {
        this.input = input;
        this.sChan = sChan;
        this.output = output;

        running = false;
    }

    @Override
    public void run() {

        running = true;

        Mat lab = new Mat();
        Mat bChan = new Mat();
        Mat sChanNorm = new Mat();
        Mat bChanNorm = new Mat();
        Mat sbChan = new Mat();
        Mat labThreshBinary = new Mat();
        Mat labThreshOtsu = new Mat();

        Imgproc.cvtColor(input,lab,Imgproc.COLOR_RGB2Lab);
        Core.extractChannel(lab,bChan,2);

        lab.release();

        Core.normalize(bChan,bChanNorm,0,255/2.0,Core.NORM_MINMAX);
        Core.normalize(sChan,sChanNorm,0,255/2.0,Core.NORM_MINMAX);

        Core.add(bChanNorm,sChanNorm,sbChan);

        bChanNorm.release();
        sChanNorm.release();;

        double stdDevMean[] = calcStdDevMean(sbChan);

        Imgproc.threshold(sbChan,labThreshBinary,stdDevMean[1]+1.5*stdDevMean[0],255,Imgproc.THRESH_BINARY);
        Imgproc.threshold(bChan,labThreshOtsu,0,255,Imgproc.THRESH_OTSU);

        bChan.release();
        sbChan.release();

        Core.bitwise_and(labThreshBinary,labThreshOtsu,output);

        running = false;

        labThreshBinary.release();
        labThreshOtsu.release();
    }

    private double[] calcStdDevMean(Mat input) {
        assert input.channels() == 1: "input must only have 1 channel"; //Makes sure image is only 1 channel (ex: black and white)

        //Calculates image mean and standard deviation
        MatOfDouble std = new MatOfDouble();
        MatOfDouble mean = new MatOfDouble();
        Core.meanStdDev(input,mean,std);
        double[] output = new double[] {std.get(0,0)[0],mean.get(0,0)[0]};

        //Removes used images from memory to avoid overflow crashes
        std.release();
        mean.release();

        //returns output data with the following format: {standard deviation, mean}
        return output;
    }

}
