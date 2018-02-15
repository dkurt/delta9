package org.delta9.testproject;  // TODO: Replace to an appropriate package.

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {

  private Net faceNet, genderNet;
  private static final String TAG = "Delta9/Test";
  private CameraBridgeViewBase mOpenCvCameraView;

  // Optionally, send frames from device to PC
  private final int SERVER_SOCKET_PORT = 43656;
  private OutputStream clientSocketOut = null;

  public void onCameraViewStarted(int width, int height) {
    // Before run, copy model onto SD card once:
    // 1. Create a folder
    //    adb shell
    //    mkdir sdcard/Android/data/org.delta9.testproject
    // 2. Copy files
    //    adb push opencv_face_detector.prototxt sdcard/Android/data/org.delta9.testproject
    //    adb push opencv_face_detector.caffemodel sdcard/Android/data/org.delta9.testproject
    String proto = "/sdcard/Android/data/org.delta9.testproject/opencv_face_detector.prototxt";
    String model = "/sdcard/Android/data/org.delta9.testproject/opencv_face_detector.caffemodel";
    faceNet = Dnn.readNetFromCaffe(proto, model);

    proto = "/sdcard/Android/data/org.delta9.testproject/gender_net.prototxt";
    model = "/sdcard/Android/data/org.delta9.testproject/gender_net.caffemodel";
    genderNet = Dnn.readNetFromCaffe(proto, model);

    Log.i(TAG, "Networks loaded successfully");

    // Start a thread to connect to a PC.
    new Thread() {
      public void run() {
        try {
          ServerSocket serverSocket = new ServerSocket(SERVER_SOCKET_PORT);
          clientSocketOut = serverSocket.accept().getOutputStream();
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }.start();
  }

  public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
    // Get a new frame
    Mat frame = inputFrame.rgba();
    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2BGR);

    // Run the face detection network.
    Mat blob = Dnn.blobFromImage(frame, 1.0, new Size(100, 100),
                                 new Scalar(104.0, 177.0, 123.0), false, false);
    faceNet.setInput(blob);
    Mat out = faceNet.forward();

    float[] data = new float[(int)out.total()];
    out.reshape(1, 1).get(0, 0, data);

    // An every detection is a vector [id, classId, confidence, left, top, right, bottom].
    for (int i = 0; i < data.length; i += 7)
    {
        float conf = data[i + 2];
        if (conf < 0.8)
            continue;

        int left = (int)(data[i + 3] * frame.cols());
        int top = (int)(data[i + 4] * frame.rows());
        int right = (int)(data[i + 5] * frame.cols());
        int bottom = (int)(data[i + 6] * frame.rows());

        left = Math.max(0, Math.min(left, frame.cols() - 1));
        right = Math.max(left, Math.min(right, frame.cols()));

        top = Math.max(0, Math.min(top, frame.rows() - 1));
        bottom = Math.max(top, Math.min(bottom, frame.rows()));

        // Crop a face from frame and pass to gender classification network.
        Mat faceImg = frame.submat(top, bottom, left, right);
        blob = Dnn.blobFromImage(faceImg, 1.0, new Size(227, 227),
                                 new Scalar(78.4263377603, 87.7689143744, 114.895847746),
                                 false, false);
        genderNet.setInput(blob);
        Mat genderOut = genderNet.forward();

        // Draw a blue bounding box around the face or a pink one depends on gender.
        float[] genderOutData = new float[2];
        genderOut.get(0, 0, genderOutData);

        Scalar bboxColor = genderOutData[0] > genderOutData[1] ?
                             new Scalar(255, 0, 0) : new Scalar(255, 0, 255);

        Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
                          bboxColor, 3);
    }
    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);

    if (clientSocketOut != null) {
      // Send to client.
      sendImg(frame);
    }
    return frame;
  }

  // Initialize OpenCV manager.
  private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
    @Override
    public void onManagerConnected(int status) {
      switch (status) {
        case LoaderCallbackInterface.SUCCESS: {
          Log.i(TAG, "OpenCV loaded successfully");
          mOpenCvCameraView.enableView();
          break;
        }
        default: {
          super.onManagerConnected(status);
          break;
        }
      }
    }
  };

  @Override
  public void onResume() {
    super.onResume();
    OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    // Set up camera listener.
    mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.CameraView);
    mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
    mOpenCvCameraView.setCvCameraViewListener(this);
  }

  public void onCameraViewStopped() {}

  // Method to send an image.
  private void sendImg(Mat img) {
    int rows = img.rows();
    int cols = img.cols();
    int channels = img.channels();
    byte[] imgData = new byte[rows * cols * channels];
    img.get(0, 0, imgData);

    byte[] data = ByteBuffer.allocate(4 * 3)
            .order(ByteOrder.nativeOrder())
            .putInt(rows).putInt(cols).putInt(channels)
            .array();
    try {
      clientSocketOut.write(data);
      clientSocketOut.write(imgData);
      clientSocketOut.flush();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
