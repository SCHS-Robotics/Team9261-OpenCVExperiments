import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class Line {
    double m;
    double b;

    double xCoord;
    boolean isVertical = false;

    Point start;
    Point end;

    double length;

    public Line(double m, double b) {
        this.m = m;
        this.b = b;
    }
    public Line(double xCoord) {
        this.xCoord = xCoord;
        this.isVertical = true;
    }
    public Line(Point a, Point b) {
        this.m = (b.y - a.y)/(b.x - a.x);
        this.b = -this.m*a.x + a.y;
        this.length = calcDist(a,b);

        start = a;
        end = b;

        if(this.m == Double.POSITIVE_INFINITY || this.m == Double.NEGATIVE_INFINITY) {
            this.isVertical = true;
            this.xCoord = this.start.x;
        }
    }
    public double calc(double x) {
        double val = !isVertical ? m*x+b : this.xCoord;
        return val;
    }
    public Point getIntersectWith(Line l) {
        if(!this.isVertical && !l.isVertical) {
            double x = (l.b - this.b) / (this.m - l.m);
            double y = (this.m * x + this.b);
            return new Point(Math.round(x), Math.round(y));
        }
        else if(this.isVertical && !l.isVertical){
            return new Point(Math.round(this.xCoord),Math.round(l.calc(this.xCoord)));
        }
        else if(!this.isVertical && l.isVertical) {
            return new Point(Math.round(l.xCoord),Math.round(this.calc(l.xCoord)));
        }
        else {
            return new Point();
        }
    }

    private double calcDist(Point a, Point b) {
        return Math.sqrt(Math.pow((a.x-b.x),2)+ Math.pow((a.y-b.y),2));
    }

    public void draw(Mat input) {
        if(!this.isVertical) {
            Imgproc.line(input,new Point(0,this.b), new Point(640,this.calc(640)),new Scalar(0,255,0),1);
        }
        else {
            Imgproc.line(input,new Point(this.xCoord,0), new Point(this.xCoord,640), new Scalar(0,255,0),1
            );
        }
    }
}
