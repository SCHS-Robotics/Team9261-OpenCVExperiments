import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NonMaxSuppressor {

    private static double thresh;

    public NonMaxSuppressor(double thresh) {
        thresh = thresh;
    }

    public static List<Rect> suppressNonMax(List<Rect> boxes) {
        if(boxes.size() == 0) {
            return boxes;
        }

        List<Integer> selectedIdxes = new ArrayList<>();

        List<Double> x1 = new ArrayList<>(); //top left x
        List<Double> y1 = new ArrayList<>(); //top left y
        List<Double> x2 = new ArrayList<>(); //bottom right x
        List<Double> y2 = new ArrayList<>(); //bottom right y

        List<Double> areas = new ArrayList<>();

        for(Rect box : boxes) {
            x1.add((double) box.x);
            y1.add((double) box.y);
            x2.add((double) box.width);
            y2.add((double) box.height);

            areas.add((x2.get(boxes.indexOf(box))-x1.get(boxes.indexOf(box))+1)*(y2.get(boxes.indexOf(box))-y1.get(boxes.indexOf(box))+1));
        }

        List<Integer> idxes = argsort(y2);

        while(idxes.size() > 0) {

            int last = idxes.size() - 1;
            int i = idxes.get(last);

            selectedIdxes.add(i);

            List<Double> xx1 = listMax(x1.get(i),x1,idxes.subList(0,last));
            List<Double> yy1 = listMax(y1.get(i),y1,idxes.subList(0,last));
            List<Double> xx2 = listMin(x2.get(i),x2,idxes.subList(0,last));
            List<Double> yy2 = listMin(y2.get(i),y2,idxes.subList(0,last));

            List<Double> w = listMax(0,addListConstant(subtractLists(xx2,xx1),1));
            List<Double> h = listMax(0,addListConstant(subtractLists(yy2,yy1),1));

            List<Double> overlap = divideLists(multiplyLists(w,h),areas,idxes.subList(0,last));

            idxes.remove(last);
            deleteBadIndexes(idxes,overlap);
        }

        return populateList(boxes,selectedIdxes);
    }

    private static List<Rect> populateList(List<Rect> boxes, List<Integer> idxes) {
        List<Rect> output = new ArrayList<>();
        for(int idx : idxes) {
            output.add(boxes.get(idx));
        }
        return output;
    }

    private static List<Double> addListConstant(List<Double> lst, double x) {
        List<Double> output = new ArrayList<>();
        for(double val : lst) {
            output.add(val+x);
        }
        return output;
    }
    private static List<Double> subtractLists(List<Double> lst1, List<Double> lst2) {
        assert lst1.size() == lst2.size(): "lists not the same size";
        List<Double> output = new ArrayList<>();
        for (int i = 0; i < lst1.size(); i++) {
            output.add(lst1.get(i)-lst2.get(i));
        }
        return output;
    }
    private static List<Double> divideLists(List<Double> lst1, List<Double> lst2, List<Integer> mask) {
        assert lst1.size() == lst2.size(): "lists not the same size";
        List<Double> output = new ArrayList<>();
        for (int i = 0; i < mask.size(); i++) {
            output.add(lst1.get(mask.get(i))/lst2.get(mask.get(i)));
        }
        return output;
    }
    private static List<Double> multiplyLists(List<Double> lst1, List<Double> lst2) {
        assert lst1.size() == lst2.size(): "lists not the same size";
        List<Double> output = new ArrayList<>();
        for (int i = 0; i < lst1.size(); i++) {
            output.add(lst1.get(i)*lst2.get(i));
        }
        return output;
    }

    private static List<Integer> deleteBadIndexes(List<Integer> idxes, List<Double> overlaps) {
        List<Integer> badIndicies = new ArrayList<>();
        for(int i = 0; i < overlaps.size(); i++) {
            if(overlaps.get(i) > thresh && !badIndicies.contains(i)) {
                badIndicies.add(i);
            }
        }
        idxes.removeAll(badIndicies);
        return idxes;
    }

    private static List<Double> listMax(double x, List<Double> lst, List<Integer> mask) {
        List<Double> output = new ArrayList<>();
        for(int idx : mask) {
            output.add(Math.max(x,lst.get(idx)));
        }
        return output;
    }
    private static List<Double> listMax(double x, List<Double> lst) {
        List<Double> output = new ArrayList<>();
        for(double val : lst) {
            output.add(Math.max(x,val));
        }
        return output;
    }
    private static List<Double> listMin(double x, List<Double> lst, List<Integer> mask) {
        List<Double> output = new ArrayList<>();
        for(int idx : mask) {
            output.add(Math.min(x,lst.get(idx)));
        }
        return output;
    }
    private static List<Double> listDivide(double cnst, List<Double> lst, List<Integer> mask) {
        List<Double> output = new ArrayList<>();
        for(int idx : mask) {
            output.add(cnst/lst.get(idx));
        }
        return output;
    }
    private static List<Integer> argsort(List<Double> input) {
        List<Integer> output = new ArrayList<>();

        List<Double> lst1 = new ArrayList<>();
        List<Double> lst2 = new ArrayList<>();

        lst1.addAll(input);
        lst2.addAll(input);

        Collections.sort(lst1);

        int multiples = 0;
        for (Double num : lst1) {
            if(!output.contains(lst2.indexOf(num))) {
                output.add(lst2.indexOf(num));
                multiples = 0;
            }
            else {
                multiples++;
                output.add(lst2.indexOf(num) + multiples);
            }
            lst2.remove(num);

        }

        return output;
    }
}
