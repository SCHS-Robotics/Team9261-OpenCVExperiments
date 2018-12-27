import org.opencv.core.Mat;

public class Key {
    public Mat image;
    public String name;

    public Key(Mat image, String name) {
        this.image = image;
        this.name = name;
    }
}
