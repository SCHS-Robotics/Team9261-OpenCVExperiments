import org.opencv.core.Core;
import org.opencv.core.Point;

public class PointState {

    public Point pt;
    public Point velocity;

    public PointState(double x, double y, double vx, double vy) {

        pt = new Point(x,y);
        velocity = new Point(vx,vy);

    }
}
