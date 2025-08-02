import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.dnn.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

public class CaffeLiveDetection {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        String proto = "models/MobileNetSSD_deploy.prototxt";
        String model = "models/MobileNetSSD_deploy.caffemodel";

        // Load the Caffe model
        Net net = Dnn.readNetFromCaffe(proto, model);

        String[] classNames = {
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        };

        VideoCapture cap = new VideoCapture(0);
        if (!cap.isOpened()) {
            System.out.println("Cannot open camera");
            return;
        }

        Mat frame = new Mat();
        while (true) {
            cap.read(frame);
            if (frame.empty()) break;

            Mat blob = Dnn.blobFromImage(frame, 0.007843, new Size(300, 300), new Scalar(127.5), false, false);
            net.setInput(blob);
            Mat detections = net.forward();

            detections = detections.reshape(1, (int)detections.total() / 7);

            for (int i = 0; i < detections.rows(); i++) {
                double confidence = detections.get(i, 2)[0];
                if (confidence > 0.5) {
                    int classId = (int)detections.get(i, 1)[0];
                    int xLeft = (int)(detections.get(i, 3)[0] * frame.cols());
                    int yTop = (int)(detections.get(i, 4)[0] * frame.rows());
                    int xRight = (int)(detections.get(i, 5)[0] * frame.cols());
                    int yBottom = (int)(detections.get(i, 6)[0] * frame.rows());

                    Imgproc.rectangle(frame, new Point(xLeft, yTop), new Point(xRight, yBottom), new Scalar(0, 255, 0), 2);
                    String label = classNames[classId] + ": " + String.format("%.2f", confidence);
                    Imgproc.putText(frame, label, new Point(xLeft, yTop - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 1);

                    System.out.println("Detected: " + label);
                }
            }

            HighGui.imshow("Caffe Live Detection", frame);
            if (HighGui.waitKey(1) == 27) break; // ESC to exit
        }

        cap.release();
        HighGui.destroyAllWindows();
    }
}

// complile javac -cp "lib/opencv-4110.jar" -d bin src/CaffeLiveDetection.java
// run java -cp "lib/opencv-4110.jar;bin" -Djava.library.path=lib CaffeLiveDetection
