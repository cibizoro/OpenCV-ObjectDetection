import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class Utils {
    public static float[] preprocess(Mat image) {
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(640, 640));
        Imgproc.cvtColor(resized, resized, Imgproc.COLOR_BGR2RGB);

        int channels = 3;
        int width = 640;
        int height = 640;
        float[] data = new float[channels * width * height];

        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double[] pixel = resized.get(y, x);
                    data[idx++] = (float) (pixel[c] / 255.0);
                }
            }
        }

        return data;
    }
}
