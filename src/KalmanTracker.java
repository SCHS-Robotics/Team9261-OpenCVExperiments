import org.opencv.core.*;
import org.opencv.video.KalmanFilter;

public class KalmanTracker {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public PointState state;

    private Point lastResult;
    private KalmanFilter kf;

    private int totalUpdates;

    public double trustworthyness;
    public KalmanTracker(Point init) {

        state = new PointState(init.x,init.y,0,0);

        //dynamParams: number of things in the state vector that change
        //measureParams: number of parameters in the state vector that are measured (x and y position only so 2)
        //control params you shouldn't worry about too much
        kf = new KalmanFilter(4, 2, 0, CvType.CV_32F);

        //transitionMatrix
        Mat transitionMatrix = new Mat(4, 4, CvType.CV_32F, new Scalar(0));
        float[] tM = {
                1, 0, 1, 0,
                0, 1, 0, 1,
                0, 0, 1, 0,
                0, 0, 0, 1 };
        transitionMatrix.put(0,0,tM);

        kf.set_transitionMatrix(transitionMatrix);

        lastResult = state.pt;
        Mat statePre = new Mat(4, 1, CvType.CV_32F, new Scalar(0)); // Toa do (x,y), van toc (0,0)
        statePre.put(0, 0, state.pt.x);
        statePre.put(1, 0, state.pt.y);
        kf.set_statePre(statePre);

        kf.set_measurementMatrix(Mat.eye(2,4, CvType.CV_32F));

        Mat processNoiseCov = Mat.eye(4, 4, CvType.CV_32F);
        processNoiseCov = processNoiseCov.mul(processNoiseCov, 1e-4);
        kf.set_processNoiseCov(processNoiseCov);

        Mat id1 = Mat.eye(2,2, CvType.CV_32F);
        id1 = id1.mul(id1,1e-1);
        kf.set_measurementNoiseCov(id1);

        Mat id2 = Mat.eye(4,4, CvType.CV_32F);
        //id2 = id2.mul(id2,0.1);
        kf.set_errorCovPost(id2);

        trustworthyness = 0;
        totalUpdates = 0;
    }

    public Point update(Point p, boolean dataCorrect) {
        Mat measurement = new Mat(2, 1, CvType.CV_32F, new Scalar(0)) ;

        totalUpdates++;

        if (!dataCorrect) {
            measurement.put(0, 0, lastResult.x);
            measurement.put(1, 0, lastResult.y);

            trustworthyness = trustworthyness/totalUpdates;
        } else {
            measurement.put(0, 0, p.x);
            measurement.put(1, 0, p.y);

            trustworthyness = (1+trustworthyness)/totalUpdates;
        }
        // Correction
        Mat estimated = kf.correct(measurement);
        lastResult.x = estimated.get(0, 0)[0];
        lastResult.y = estimated.get(1, 0)[0];
        return lastResult;
    }

    public Point getPrediction() {
        Mat prediction = kf.predict();
        lastResult = new Point(prediction.get(0, 0)[0], prediction.get(1, 0)[0]);
        return lastResult;
    }
    public Point correction(Point p){
        Mat measurement = new Mat(2, 1, CvType.CV_32F, new Scalar(0));
        measurement.put(0, 0, p.x);
        measurement.put(1, 0, p.y);

        Mat estimated = kf.correct(measurement);
        lastResult.x = estimated.get(0, 0)[0];
        lastResult.y = estimated.get(1, 0)[0];
        return lastResult;
    }
}
