import org.opencv.core.Rect;

public class Detection {
    public Rect box;
    public String label;
    public float confidence;

    public Detection(Rect box, String label, float confidence) {
        this.box = box;
        this.label = label;
        this.confidence = confidence;
    }
}
