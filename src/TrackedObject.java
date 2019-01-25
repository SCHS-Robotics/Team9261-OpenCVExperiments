import org.opencv.core.Point;
import org.opencv.core.Rect;

public class TrackedObject {

    Rect r;
    KalmanTracker k;
    Point center;

    public TrackedObject(Rect r, KalmanTracker k) {
        this.r = r;
        this.k = k;
        this.center = new Point(r.x+r.width/2.0,r.y+r.height/2.0);
    }
}
