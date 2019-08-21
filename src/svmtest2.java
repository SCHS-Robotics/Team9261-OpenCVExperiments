import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.LogisticRegression;
import org.opencv.ml.Ml;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.xfeatures2d.SIFT;
import org.opencv.xfeatures2d.SURF;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class svmtest2 {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String[] args){

        String DATABASE = "C:\\TrainingArena\\OCR\\Training";

        //List initialization
        ArrayList<Integer> training_labels_array = new ArrayList<>();
        ArrayList<Integer> testing_labels_array = new ArrayList<>();
        Mat TRAINING_DATA = new Mat();
        Mat TESTING_DATA = new Mat();

        // Load training and testing data
        File[] files = new File(DATABASE).listFiles();

        HOGDescriptor hog = new HOGDescriptor(new Size(320,240),new Size(40,30),new Size(20,15), new Size(40,30),16);

        SIFT sift = SIFT.create();

        int numImages = 250/2;
        int negLabel = 0;
        int posLabel = 1;
        Size winStride = new Size(128,128);
        Size padding = new Size(8,8);

        loadData(files,hog,posLabel,TRAINING_DATA,training_labels_array,numImages,winStride,padding,0);
        loadData(files,hog,negLabel,TRAINING_DATA,training_labels_array,numImages,winStride,padding,0);

        TRAINING_DATA = TRAINING_DATA.reshape(0,2*numImages);
        TESTING_DATA = TESTING_DATA.reshape(0,2*numImages);

        // Put training and testing labels into Mats
        Mat TRAINING_LABELS = Mat.zeros(TRAINING_DATA.rows(),1, CvType.CV_32SC1);
        for(int i = 0; i < training_labels_array.size(); i++){
            TRAINING_LABELS.put(i, 0, training_labels_array.get(i));
        }
        Mat TESTING_LABELS = Mat.zeros(TESTING_DATA.rows(), 1, CvType.CV_32SC1);
        for(int i = 0; i < testing_labels_array.size(); i++){
            TESTING_LABELS.put(i, 0, testing_labels_array.get(i));
        }

        TRAINING_LABELS.convertTo(TRAINING_LABELS,CvType.CV_32F);
        TESTING_LABELS.convertTo(TESTING_LABELS,CvType.CV_32F);


        //TRAINING_LABELS = TRAINING_LABELS.reshape(0,(int) hogSize.height);

        System.out.println("TRAINING_DATA - Rows:" + TRAINING_DATA.rows() + " Cols:" + TRAINING_DATA.cols());
        System.out.println("TRAINING_LABELS - Rows:" + TRAINING_LABELS.rows() + " Cols:" + TRAINING_LABELS.cols());
        //System.out.println(TRAINING_LABELS.dump());
        System.out.println("TESTING_DATA - Rows:" + TESTING_DATA.rows() + " Cols:" + TESTING_DATA.cols());
        System.out.println("TESTING_LABELS - Rows:" + TESTING_LABELS.rows() + " Cols:" + TESTING_LABELS.cols());
        //System.out.println(TRAINING_LABELS.dump());


        // Train SVM
        LogisticRegression log = LogisticRegression.create();

        log.setRegularization(LogisticRegression.REG_L1);
        log.setTrainMethod(LogisticRegression.BATCH);
        log.setMiniBatchSize(100);
        log.setIterations(1000);
        log.setLearningRate(0.0001);

        /*SVM svm = SVM.create();
        svm.setKernel(SVM.LINEAR);
        svm.setType(SVM.C_SVC);
        svm.setC(100);
        svm.setGamma(3);*/
        // errors here
        log.train(TESTING_DATA, Ml.ROW_SAMPLE, TESTING_LABELS);

        System.out.println("Train Error: "+Double.toString(100*calcErrorLog(log,TRAINING_DATA,TRAINING_LABELS))+"%");
        System.out.println("Test Error: "+Double.toString(100*calcErrorLog(log,TESTING_DATA,TESTING_LABELS))+"%");
    }

    private static double calcErrorLog(LogisticRegression l,Mat testData,Mat labels) {
        int wrong = 0;
        for(int i = 0; i< testData.rows(); i++) {
            Mat check = testData.row(i);
            check = check.reshape(1, 1);
            float label = l.predict(check);
            if(label != labels.get(i,0)[0]) {
                wrong++;
            }
        }
        return (1.0*wrong)/(1.0*testData.rows());
    }

    private static void loadData(File[] directory, HOGDescriptor hog, int label, Mat data, List<Integer> labels_array, int numImages, Size winStride, Size padding, int start) {
        for(int i = start; i < start+numImages; i++){

            Mat image = Imgcodecs.imread(directory[i].getAbsolutePath(),Imgcodecs.IMREAD_UNCHANGED); //for each file, read the image

            System.out.println(directory[i].getName());

            Mat training_feature = new MatOfFloat();
            MatOfPoint locations = new MatOfPoint();
            MatOfFloat a = new MatOfFloat();
            training_feature = getDescriptors(image);
            //hog.compute(image,a,winStride,padding,locations);
            //System.out.println(a.size());
            locations.release();
            data.push_back(training_feature);

            //System.out.println(training_feature.dump());

            training_feature.release();

            labels_array.add(label);
            System.out.println(Math.round(100*100*(((i+1)*1.0)/(start+numImages*1.0)))/100.0+"%");
            image.release();
            System.gc();
        }
    }

    private static Mat getDescriptors(Mat image) {
        SURF detector = SURF.create();

        MatOfKeyPoint kp = new MatOfKeyPoint();

        detector.detect(image,kp);
        SURF extractor = SURF.create(); //SURF, someday...;

        Mat desc = new MatOfFloat();

        extractor.compute(image, kp, desc);
        kp.release();
        image.release();

        //
        // desc = desc.reshape(1,(int) (desc.size().width*desc.size().height));

        System.out.println(desc.size());

        return desc;
    }

    private static String getFileExtension(File file) {
        String name = file.getName();
        int lastIndexOf = name.lastIndexOf(".");
        if (lastIndexOf == -1) {
            return ""; // empty extension
        }
        return name.substring(lastIndexOf);
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
