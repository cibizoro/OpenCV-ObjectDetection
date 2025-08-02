import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import java.util.List;
import java.io.File;
import javax.sound.sampled.*;

public class Main {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void playAlertSound() {
        try {
            File soundFile = new File("C:\\Users\\CIBI ARULNATH\\cibicv\\cibicv9\\cibicv9\\src\\sounds\\alert.wav");
            if (!soundFile.exists()) {
                System.out.println("❌ Sound file not found!");
                return;
            }
            AudioInputStream audioIn = AudioSystem.getAudioInputStream(soundFile);
            Clip clip = AudioSystem.getClip();
            clip.open(audioIn);
            clip.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception {
        VideoCapture cap = new VideoCapture(0);
        if (!cap.isOpened()) {
            System.out.println("❌ Cannot open camera.");
            return;
        }

        ObjectDetector detector = new ObjectDetector("models/modified_model.onnx");
        Mat frame = new Mat();

        while (true) {
            long startTime = System.nanoTime();
            cap.read(frame);
            if (frame.empty()) continue;

            // Define ROI: top-left = (160, 60), bottom-right = (480, 380)
            int roiX = 160;
            int roiY = 60;
            int roiWidth = 320;
            int roiHeight = 320;
            Rect roi = new Rect(roiX, roiY, roiWidth, roiHeight);

            // Draw ROI rectangle
            Imgproc.rectangle(frame,
                    new Point(roiX, roiY),
                    new Point(roiX + roiWidth, roiY + roiHeight),
                    new Scalar(255, 255, 255), 2);

            List<Detection> detections = detector.detect(frame);

            for (Detection d : detections) {
                if (d.confidence < 0.45) continue;

                if (!rectContains(roi, d.box)) continue;

                // Draw bounding box
                Imgproc.rectangle(frame, d.box, new Scalar(0, 255, 0), 2);

                // Label text and coordinates
                String labelText = d.label + " " + String.format("%.1f", d.confidence * 100) + "%";
                int baseLine[] = new int[1];
                Size labelSize = Imgproc.getTextSize(labelText, Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, 2, baseLine);

                int labelX = d.box.x;
                int labelY = (d.box.y - 10 < labelSize.height)
                        ? (int)(d.box.y + labelSize.height + 10)
                        : (int)(d.box.y - 10);

                // Draw black background for text
                Imgproc.rectangle(frame,
                        new Point(labelX, labelY - labelSize.height),
                        new Point(labelX + labelSize.width, labelY + baseLine[0]),
                        new Scalar(0, 0, 0), Core.FILLED);

                // Draw label text in yellow
                Imgproc.putText(frame, labelText,
                        new Point(labelX, labelY),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 255, 255), 2);

                // Play alert if person detected
                if (d.label.equalsIgnoreCase("person")) {
                    playAlertSound();
                }
            }

            // Show FPS
            double fps = 1e9 / (System.nanoTime() - startTime);
            Imgproc.putText(frame, "FPS: " + String.format("%.2f", fps),
                    new Point(10, 25), Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.6, new Scalar(0, 255, 255), 2);

            HighGui.imshow("Object Detection – Labels, ROI & Alert", frame);
            if (HighGui.waitKey(1) == 'q') break;
        }

        cap.release();
        HighGui.destroyAllWindows();
    }

    // Helper: check if a Rect (inner) is completely inside another (outer)
    public static boolean rectContains(Rect outer, Rect inner) {
        return inner.x >= outer.x && inner.y >= outer.y &&
                inner.x + inner.width <= outer.x + outer.width &&
                inner.y + inner.height <= outer.y + outer.height;
    }
}