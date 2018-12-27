import org.opencv.core.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class KMedioids {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String args[]) {
        //List<List<Integer>> data = Arrays.asList(Arrays.asList(2,6),Arrays.asList(3,4),Arrays.asList(3,8),Arrays.asList(4,7),Arrays.asList(6,2),Arrays.asList(6,4),Arrays.asList(7,3),Arrays.asList(7,4),Arrays.asList(8,5),Arrays.asList(7,6));
        int points[][] = {{2,6},{3,4},{3,8},{4,7},{6,2},{6,4},{7,3},{7,4},{8,5},{7,6}};

        Mat data = cvt2Mat(points);
        Mat medioids = initMedioids(data,2);

        Mat dmat = getDistanceMatrix(data,medioids);

        ArrayList<Mat> clusters = new ArrayList<>();
        int cost = cluster(data,dmat,clusters);
        int prevCost = cost + 1;
        /*
        Initialize: select k of the n data points as the medoids --done
        Associate each data point to the closest medoid.
        While the cost of the configuration decreases:
            For each medoid m, for each non-medoid data point o:
                Swap m and o, recompute the cost (sum of distances of points to their medoid)
                If the total cost of the configuration increased in the previous step, undo the swap
         */

        while(prevCost-cost > 0) {
            prevCost = cost;
            for (int i = 0; i < medioids.rows(); i++) {
                for (int j = 0; j < pickNonMedioids(data,medioids).rows(); j++) {
                    //swap medioid and random data point
                    if(1==1) { //if swap increased cost
                        //undo swap
                    }
                }
            }

        }

        medioids = pickNonMedioids(data,medioids);
        System.out.println(medioids.dump());
        /*clusters.clear();
        int cost = cluster(data,getDistanceMatrix(data,medioids),clusters);

        for (int i = 0; i < medioids.rows(); i++) {
            int tries = 0;
            while(cost >= initCost && tries < data.rows()) {
                medioids = pickNonMedioids(data,medioids);
                clusters.clear();
                cost = cluster(data,getDistanceMatrix(data,medioids),clusters);
                tries++;
            }
        }*/
        //System.out.println(medioids.dump());
        //System.out.println(cost);
        //System.out.println(clusters.get(0).dump());
        //System.out.println(clusters.get(1).dump());
    }

    private static Mat pickNewMedioids(Mat data, Mat medioids) {
        Random rand = new Random();
        int replacedMedioidIdx = rand.nextInt(medioids.rows());
        int newMedioidIdx = rand.nextInt(data.rows());
        while(matsAreEqual(medioids.row(replacedMedioidIdx),data.row(newMedioidIdx))) {
            newMedioidIdx = rand.nextInt(data.rows());
        }
        Mat newMedioids = new Mat();
        newMedioids.push_back(data.row(newMedioidIdx));
        for (int medioidIdx = 0; medioidIdx < medioids.rows(); medioidIdx++) {
            if(!matsAreEqual(medioids.row(replacedMedioidIdx),medioids.row(medioidIdx))) {
                newMedioids.push_back(medioids.row(medioidIdx));
            }
        }
        medioids.release();
        System.gc();
        return newMedioids;
    }
    private static Mat pickNonMedioids(Mat data, Mat medioids) {
        Mat nonMedioids = new Mat();
        ArrayList<Mat> used = new ArrayList<>();
        for (int i = 0; i < medioids.rows(); i++) {
            used.add(medioids.row(i));
        }
        for (int dataIdx = 0; dataIdx < data.rows(); dataIdx++) {
            for (int medioidIdx = 0; medioidIdx < medioids.rows(); medioidIdx++) {
                if(!matsAreEqual(medioids.row(medioidIdx),data.row(dataIdx)) && !containsMat(used,data.row(dataIdx))) {
                    if (used.size() > 0) {
                        System.out.println("medioid: "+medioids.row(medioidIdx).dump());
                        displayAllMats(used);
                        System.out.println(data.row(dataIdx).dump());
                        System.out.println(matsAreEqual(medioids.row(medioidIdx),data.row(dataIdx)));
                        System.out.println();

                    }
                    nonMedioids.push_back(data.row(dataIdx));
                    used.add(data.row(dataIdx));

                }
            }
        }
        System.out.println();
        System.out.println(medioids.dump());
        used.clear();
        return nonMedioids;
    }

    private static void displayAllMats(ArrayList<Mat> a) {
        for (Mat mat: a) {
            System.out.println(mat.dump());
        }
    }

    private static boolean containsMat(ArrayList<Mat> array, Mat mat) {
        boolean matched = false;
        for(Mat element : array) {
            if(matsAreEqual(mat,element)) {
                matched = true;
            }
        }
        return matched;
    }

    private static int cluster(Mat data,Mat distanceMatrix,ArrayList<Mat> clusters) {
        int totalCost = 0;
        for (int i = 0; i < distanceMatrix.rows(); i++) {
            clusters.add(new Mat());
        }
        for(int pointidx = 0; pointidx < data.rows(); pointidx++) {
            Core.MinMaxLocResult minMax = Core.minMaxLoc(distanceMatrix.row(pointidx));
            totalCost += minMax.minVal;
            clusters.get((int) minMax.minLoc.x).push_back(data.row(pointidx));
        }
        distanceMatrix.release();
        System.gc();
        return totalCost;
    }

    private static Mat getDistanceMatrix(Mat data, Mat medioids) {
        Mat distanceMatrix = new Mat(data.rows(),medioids.cols(), CvType.CV_32SC1); //see wikipedia for how matrix is formatted (look up K-medioids)
        for(int medioididx = 0; medioididx < medioids.rows(); medioididx++) {
            for(int rowidx = 0; rowidx < data.rows(); rowidx++) {
                int dist = calcDist(medioids.row(medioididx),data.row(rowidx));
                distanceMatrix.put(rowidx,medioididx, new int[] {dist});
            }
        }
        return distanceMatrix;
    }

    private static Mat initMedioids(Mat points,int numClusters) {
        assert numClusters <= (int) points.size().height : "Cluster number bigger than the dataset";
        assert numClusters > 0 : "Cluster number must be greater than 0";
        ArrayList<Integer> idxes = new ArrayList<>();
        Mat pts = new Mat();
        Random rand = new Random();
        for(int i = 0; i < numClusters; i++) {
            int idx = rand.nextInt(points.rows());
            Mat pt = points.row(idx);

            while(idxes.contains(idx)) {
                idx = rand.nextInt(points.rows());
                pt = points.row(idx);
            }
            idxes.add(idx);
            pts.push_back(pt);
            pt.release();

        }
        idxes.clear();
        System.gc();
        return pts;
    }

    //Calculates manhattan distance between two points
    private static int calcDist(Mat a, Mat b) {
        assert a.size().width == b.size().width : "Sizes of a and b are not equal";
        assert a.rows() == 1 && b.rows() == 1 : "Inputs must be two points (only 1 row)";
        int dist = 0;
        for(int i = 0; i < a.cols(); i++) {
            dist += Math.abs(a.get(0,i)[0]-b.get(0,i)[0]);
        }
        return dist;
    }

    private static boolean matsAreEqual(Mat a, Mat b) {
        return Core.norm(a,b,Core.NORM_L1) == 0;
    }

    private static Mat cvt2Mat(int ints[][]) {
        Mat mat = new Mat(ints.length,ints[0].length,CvType.CV_32SC1);
        for (int row = 0; row < ints.length; row++) {
            mat.put(row, 0, ints[row]);
        }
        return mat;
    }
}
