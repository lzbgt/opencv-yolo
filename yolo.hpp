/**
 * Author: bruce.lu (lzbgt@icloud.com)
 */

#ifndef _MY_YOLO_HPP_
#define _MY_YOLO_HPP_

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>

#include "spdlog/spdlog.h"

#ifdef _MY_HEADERS_
#include <opencv2/core/types_c.h>

#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#else
#include <opencv2/core/types_c.h>

#include <opencv2/opencv.hpp>
#endif

using namespace cv;
using namespace dnn;
using namespace std;

class YoloDectect {
   public:
    unsigned long numFrameProcessed = 0;

   private:
    // Initialize the parameters
    const string selfId = "ObjectDetector";
    float confThreshold = 0.1f;  // Confidence threshold
    float nmsThreshold = 0.2f;   // Non-maximum suppression threshold
    int inpWidth = 416;          // Width of network's input image
    int inpHeight = 416;         // Height of network's input image
    vector<string> classes;
    Net net;
    Mat blob;
    VideoCapture cap;
    VideoWriter video;
    bool bOutputIsImg = false;
    bool bInputIsImage = true;
    string outFileBase;
    bool cmdStop = false;
    unsigned int wrapNum = 0;
    unsigned int numLogSkip = 0;
    bool bHumanOnly = false;
    bool bContinue = true;
    int cameNo = -1;

    // Get the names of the output layers
    vector<String> getOutputsNames(const Net& net) {
        static vector<String> names;
        if (names.empty()) {
            //Get the indices of the output layers, i.e. the layers with unconnected outputs
            vector<int> outLayers = net.getUnconnectedOutLayers();

            //get the names of all the layers in the network
            vector<String> layersNames = net.getLayerNames();

            // Get the names of the output layers in names
            names.resize(outLayers.size());
            for (size_t i = 0; i < outLayers.size(); ++i)
                names[i] = layersNames[outLayers[i] - 1];
        }
        return names;
    }

    // post process
    bool postprocess(Mat& frame, const vector<Mat>& outs, bool bModify = false) {
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;
        bool found = false;

        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                // Get the value and location of the maximum score
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold) {
                    if (classIdPoint.x == 0) {
                        found = true;
                        break;
                    }
                }
            }
            if (found) {
                break;
            }
        }

        return found;
    }

    //
   protected:
    //
   public:
    typedef int (*callback)(vector<tuple<string, double, Rect>>&, Mat);
    YoloDectect(string path = ".", bool _humanOnly = false, float confThresh = 0.1, bool _bContinue = true, unsigned int _wrapNum = 10, unsigned int _numLogSkip = 380) {
        if (path.empty()) {
            path = ".";
        }

        bHumanOnly = _humanOnly;
        bContinue = _bContinue;

        confThreshold = confThresh;

        wrapNum = _wrapNum;
        numLogSkip = _numLogSkip;

        // Load names of classes
        string classesFile = path + "/coco.names";
        // Give the configuration and weight files for the model
        String modCfg = path + "/yolov3-tiny.cfg";
        String modWeights = path + "/yolov3-tiny.weights";

        // if(!fs::exists(classesFile) || !fs::exists(modCfg) || !fs::exists(modWeights)) {
        //     spdlog::error("{} failed to load configration files", selfId);
        //     exit(1);
        // }

        ifstream ifs(classesFile.c_str());
        string line;
        while (getline(ifs, line)) {
            classes.push_back(line);
        }

        // Load the network
        net = readNetFromDarknet(modCfg, modWeights);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        spdlog::debug("{} inited", selfId);
    }

    bool process(Mat& inFrame, Mat* pOutFrame = nullptr, bool bModify = false) {
        if (inFrame.empty()) {
            false;
        }

        // Create a 4D blob from a frame.
        blobFromImage(inFrame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

        //Sets the input to the network
        net.setInput(blob);

        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        // Remove the bounding boxes with low confidence

        auto ret = postprocess(inFrame, outs, false);

        // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        if (numLogSkip == 0 || numFrameProcessed % numLogSkip == 0) {
            double freq = getTickFrequency() / 1000;
            double t = net.getPerfProfile(layersTimes) / freq;
            spdlog::debug("{} infer time: {} ms", selfId, t);
        }

        numFrameProcessed++;
        return ret;
    }

    int process(string inVideoUri, string outFile = "processed.jpg", callback cb = nullptr) {
        if (inVideoUri.empty()) {
            inVideoUri = "0";
        }

        try {
            if (inVideoUri.substr(0, 4) == "rtsp" || inVideoUri.substr(0, 4) == "rtmp" || inVideoUri.substr(inVideoUri.find_last_of(".") + 1) == "mp4" || (cameNo = stoi(inVideoUri)) >= 0) {
                bInputIsImage = false;
            }
        } catch (...) {
        }

        if (!bInputIsImage) {
            if ((cameNo == -1 && !cap.open(inVideoUri, CAP_ANY)) || (cameNo != -1 && !cap.open(cameNo))) {
                spdlog::error("{} failed to open input video {}", selfId, inVideoUri);
                exit(1);
            }
        }

        EventDetection::getInstance().run();

        unsigned long frameCnt = 0;
        unsigned long detCnt = 0, skipCnt = 0;
        Mat frame;
        auto lastTs = std::chrono::system_clock::now();
        while (true) {
            if (!cap.isOpened()) {
                cap.open(cameNo);
            }

            if (!cap.read(frame)) {
                spdlog::error("failed read stream");
                this_thread::sleep_for(chrono::seconds(2));
                continue;
            }

            // Stop the program if reached end of video
            if (frame.empty()) {
                spdlog::error("empty");
                continue;
            }

            frameCnt++;

            bool ret = process(frame);
            if (frameCnt % 100 == 0) {
                spdlog::debug("det cnt {}, non cnt {}", detCnt, skipCnt);
            }
            if (ret == false) {
                skipCnt++;
            } else {
                detCnt++;
            }

            auto deltaS = std::chrono::duration<double>(std::chrono::system_clock::now() - lastTs).count();
            if (deltaS > 2) {
                if (detCnt > skipCnt) {
                    EventDetection::getInstance().notify(1);
                }
            }

            if (deltaS > 8) {
                spdlog::debug("8s");
                if (detCnt < skipCnt) {
                    EventDetection::getInstance().notify(0);
                }
                detCnt = 0;
                skipCnt = 0;
                lastTs = std::chrono::system_clock::now();
            }

            frame.release();
            this_thread::sleep_for(chrono::seconds(1));
        }

        spdlog::info("{} done processing {}", selfId, inVideoUri);
        cap.release();
        if (!bOutputIsImg) video.release();

        return 0;
    }
};

#endif