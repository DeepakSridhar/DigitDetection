package com.sridhar.deepak.digitdetection;

/**
 * Created by deep1 on 1/11/2018.
 */

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.flurgle.camerakit.CameraListener;
import com.flurgle.camerakit.CameraView;

import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class HappyFaceDetector extends AppCompatActivity {

    private static final int INPUT_SIZE = 64;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128;
    private static final String INPUT_NAME = "input_1";
    private static final String OUTPUT_NAME = "fc/Sigmoid";

    private static final String MODEL_FILE = "file:///android_asset/happy_model.pb";
    private static final String LABEL_FILE =
            "file:///android_asset/face_label.txt";

    private Classifier classifier;
    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult;
    private Button btnDetectObject, btnToggleCamera;
    private ImageView imageViewResult;
    private CameraView cameraView;
    private TextToSpeech textspeech;
    private int[] intValues = new int[INPUT_SIZE*INPUT_SIZE];
    private float[] floatValues = new float[INPUT_SIZE*INPUT_SIZE*3];
    private float imageStd = 128;
    private String YES = "Happy Face";
    private String NONE = "Not a Happy Face";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.catsvsdogs);
        cameraView = (CameraView) findViewById(R.id.cameraView);
        imageViewResult = (ImageView) findViewById(R.id.imageViewResult);
        textViewResult = (TextView) findViewById(R.id.textViewResult);
        textViewResult.setMovementMethod(new ScrollingMovementMethod());

        btnToggleCamera = (Button) findViewById(R.id.btnToggleCamera);
        btnDetectObject = (Button) findViewById(R.id.btnDetectObject);

        textspeech=new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != TextToSpeech.ERROR) {
                    textspeech.setLanguage(Locale.US);
                }
            }
        });

        cameraView.setCameraListener(new CameraListener() {
            @Override
            public void onPictureTaken(byte[] picture) {
                super.onPictureTaken(picture);

                Bitmap bitmap = BitmapFactory.decodeByteArray(picture, 0, picture.length);

                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

                imageViewResult.setImageBitmap(bitmap);

                bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
                for (int i = 0; i < intValues.length; ++i) {
                    final int val = intValues[i];
                    floatValues[i * 3 + 0] = (((val >> 16) & 0xFF))/imageStd;
                    floatValues[i * 3 + 1] = (((val >> 8) & 0xFF))/imageStd;
                    floatValues[i * 3 + 2] = ((val & 0xFF))/imageStd;
                    floatValues[i * 3 + 0] = floatValues[i * 3 + 0] - 1;
                    floatValues[i * 3 + 1] = floatValues[i * 3 + 1] - 1;
                    floatValues[i * 3 + 2] = floatValues[i * 3 + 2] - 1;

                final List<Classifier.Recognition> results = classifier.recognizeImage(floatValues,2);

                textViewResult.setText(results.toString());

                Toast.makeText(getApplicationContext(), results.toString(), Toast.LENGTH_SHORT).show();
                if (results.get(0).toString().equals("[1]")) {
                    textspeech.speak(YES, TextToSpeech.QUEUE_FLUSH, null);
                } else {
                    textspeech.speak(results.toString(), TextToSpeech.QUEUE_FLUSH, null);
                    textspeech.speak(NONE, TextToSpeech.QUEUE_FLUSH, null);

                }
            }
        });

        btnToggleCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.toggleFacing();
            }
        });

        btnDetectObject.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.captureImage();
            }
        });

        initTensorFlowAndLoadModel();
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraView.start();
    }

    @Override
    protected void onPause() {
        cameraView.stop();
        if(textspeech !=null){
            textspeech.stop();
        }
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        textspeech.shutdown();
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            INPUT_SIZE,
                            INPUT_NAME,
                            OUTPUT_NAME);
                    makeButtonVisible();
                    Log.d(TAG, "Load Success");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                btnDetectObject.setVisibility(View.VISIBLE);
            }
        });
    }
}
