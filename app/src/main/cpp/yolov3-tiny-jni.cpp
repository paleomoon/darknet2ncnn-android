#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>

#include <sys/time.h>
#include <unistd.h>

#include <stdio.h>
#include <algorithm>
#include <fstream>

#include "darknet2ncnn.h"
#include "ncnn/src/layer/input.h"
#include "ncnn_tools.h"

extern "C" {

static CustomizedNet yolo;

struct Object
{
  cv::Rect_<float> rect;
  int label;
  float prob;
};

JNIEXPORT jboolean JNICALL
Java_com_tarmac_yolov2Tiny_yolov2Tiny_Init(JNIEnv *env, jobject obj, jstring param, jstring bin) {
    __android_log_print(ANDROID_LOG_DEBUG, "yolov2TinyJni", "enter the jni func");

     register_darknet_layer(yolo);

    const char *param_path = env->GetStringUTFChars( param, NULL);
    if(param_path == NULL)
        return JNI_FALSE;
    __android_log_print(ANDROID_LOG_DEBUG, "yolov2TinyJni", "load_param %s", param_path);

    int ret = yolo.load_param(param_path);
    __android_log_print(ANDROID_LOG_DEBUG, "yolov2TinyJni", "load_param result %d", ret);
    env->ReleaseStringUTFChars( param, param_path);

    const char *bin_path = env->GetStringUTFChars( bin, NULL);
    if(bin_path == NULL)
        return JNI_FALSE;
    __android_log_print(ANDROID_LOG_DEBUG, "yolov2TinyJni", "load_model %s", bin_path);

    int ret2 = yolo.load_model(bin_path);
    __android_log_print(ANDROID_LOG_DEBUG, "yolov2TinyJni", "load_model result %d", ret2);
    env->ReleaseStringUTFChars( bin, bin_path);
    return JNI_TRUE;
}

JNIEXPORT jfloatArray JNICALL Java_com_tarmac_yolov2Tiny_yolov2Tiny_Detect(JNIEnv* env, jobject thiz, jobject bitmap)
{
    ncnn::Input *input = (ncnn::Input *)yolo.get_layer_from_name("data");

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    void* indata;
    AndroidBitmap_lockPixels(env, bitmap, &indata);

//    const char *img_path = env->GetStringUTFChars( imgPath, NULL);
//    if(img_path == NULL)
//        return JNI_FALSE;
//    __android_log_print(ANDROID_LOG_DEBUG, "yolov2TinyJni", "load_img %s", img_path);
//
//    cv::Mat m = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
//    env->ReleaseStringUTFChars( imgPath, img_path);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2RGB, width, height, input->w, input->h);
    //ncnn::Mat in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_BGR2RGB, m.cols, m.rows, input->w, input->h);

    __android_log_print(ANDROID_LOG_DEBUG, "yolov2TinyJniIn", "yolov2_predict_has_input1, in.w: %d; in.h: %d", in.w, in.h);
    AndroidBitmap_unlockPixels(env, bitmap);

    const float norm_vals[3] = {1 / 255.0, 1 / 255.0, 1 / 255.0};
    in.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("data",in);
    ex.set_light_mode(false);
    ex.set_num_threads(4);

    ncnn::Mat out;
    ncnn::Blob *out_blob = yolo.get_last_layer_output_blob();
    int result = ex.extract(out_blob->name.c_str(), out);
    __android_log_print(ANDROID_LOG_DEBUG, "yolov2TinyJni", "extract stop %d", result);
   if (result != 0)
        return NULL;
    int output_wsize = out.w;
    int output_hsize = out.h;

    jfloat *output[output_wsize * output_hsize];
    for(int i = 0; i< out.h; i++) {
        for (int j = 0; j < out.w; j++) {
            output[i*output_wsize + j] = &out.row(i)[j];
        }
    }
    jfloatArray jOutputData = env->NewFloatArray(output_wsize);
    if (jOutputData == nullptr) return nullptr;
    env->SetFloatArrayRegion(jOutputData, 0,  output_wsize * output_hsize,
                             reinterpret_cast<const jfloat *>(*output));  // copy

    return jOutputData;

}
}
