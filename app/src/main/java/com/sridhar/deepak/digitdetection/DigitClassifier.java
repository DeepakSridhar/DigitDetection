package com.sridhar.deepak.digitdetection;

/**
 * Created by deep1 on 1/7/2018.
 */

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
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

public class DigitClassifier extends AppCompatActivity {

    private static final String TAG = "DigitClassifier";

    private static final int INPUT_SIZE = 28;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "conv2d_1_input";
    private static final String OUTPUT_NAME = "dense_2/Softmax";

    private static final String MODEL_FILE = "file:///android_asset/opt_mnist_convnet-keras.pb";
    private static final String LABEL_FILE =
            "file:///android_asset/graph_label_strings.txt";

    private Classifier classifier;
    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult;
    private Button btnDetectObject, btnToggleCamera;
    private ImageView imageViewResult;
    private CameraView cameraView;
    private TextToSpeech textspeech;
    private String NONE = "Unable to Detect";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.digitclassifier);
        cameraView = (CameraView) findViewById(R.id.cameraView2);
        imageViewResult = (ImageView) findViewById(R.id.imageViewResult2);
        textViewResult = (TextView) findViewById(R.id.textViewResult2);
        textViewResult.setMovementMethod(new ScrollingMovementMethod());

        btnToggleCamera = (Button) findViewById(R.id.btnToggleCamera2);
        btnDetectObject = (Button) findViewById(R.id.btnDetectObject2);

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

                /*Bitmap bwBitmap = Bitmap.createBitmap( bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.RGB_565 );
                float[] hsv = new float[ 3 ];
                for( int col = 0; col < bitmap.getWidth(); col++ ) {
                    for( int row = 0; row < bitmap.getHeight(); row++ ) {
                        Color.colorToHSV( bitmap.getPixel( col, row ), hsv );
                        if( hsv[ 2 ] > 0.5f ) {
                            bwBitmap.setPixel( col, row, 0xffffffff );
                        } else {
                            bwBitmap.setPixel( col, row, 0xff000000 );
                        }
                    }
                }*/
                Bitmap bwBitmap = convertimage(bitmap);

                imageViewResult.setImageBitmap(bwBitmap);

                int width = bwBitmap.getWidth();
                int height = bwBitmap.getHeight();

                int[] pixels = new int[width * height];
                bwBitmap.getPixels(pixels, 0, width, 0, 0, width, height);

                Log.d(TAG, "Pixel Dimension: " + pixels.length);

                float[] retPixels = new float[pixels.length];
                for (int i = 0; i < pixels.length; ++i) {
                    // Set 0 for white and 255 for black pixel
                    int pix = pixels[i];
                    int b = pix & 0xff;
                    retPixels[i] = 0xff - b;
                }

                final List<Classifier.Recognition> results = classifier.recognizeImage(retPixels,1);


                if (results.size() > 0) {
                    String value = " Number is : " +results.get(0).getTitle();
                    textViewResult.setText(value);
                    Toast.makeText(getApplicationContext(), value,Toast.LENGTH_SHORT).show();
                    textspeech.speak(value, TextToSpeech.QUEUE_FLUSH, null);
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

    public static Bitmap convertimage(Bitmap original) {
        Bitmap finalimage = Bitmap.createBitmap(original.getWidth(),original.getHeight(),original.getConfig());
        int A,R,G,B;
        int colorpixel;
        int width = original.getWidth();
        int height = original.getHeight();

        for (int x=0;x<height;x++) {
            for (int y=0;y<width;y++) {
                colorpixel = original.getPixel(x,y);
                A=Color.alpha(colorpixel);
                R=Color.red(colorpixel);
                G=Color.green(colorpixel);
                B=Color.blue(colorpixel);

                R=(R+G+B)/3;
                G=R;
                B=R;

                finalimage.setPixel(x,y,Color.argb(A,R,G,B));

            }
        }
        return finalimage;
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
