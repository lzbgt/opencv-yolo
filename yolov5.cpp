#include <opencv2/core/types_c.h>

#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace dnn;
using namespace std;

constexpr float CONFIDENCE_THRESHOLD = 0;
constexpr float NMS_THRESHOLD = 0.4;
//number of classes to detect
//constexpr int NUM_CLASSES = 80;
constexpr int NUM_CLASSES = 5;  // to detect only one class -> the first in the coco_names_txt file list ?!??
// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

int main(int argc, char* argv) {
    cout << CV_VERSION << endl;
    cv::Mat im_1;

    im_1 = cv::imread("im_14_RGB.jpg", cv::IMREAD_COLOR);
    if (!im_1.data) {
        cout << "\n\t Could not open or find the image 1" << endl;
    }
    // let's downscale the image using new  width and height
    int down_width = 640;
    int down_height = 640;

    //resize down
    cv::resize(im_1, im_1, cv::Size(down_width, down_height), cv::INTER_LINEAR);

    // YOLO V5
    // read coco class names do ficheiro .txt
    std::vector<std::string> class_names;
    {
        std::ifstream class_file("coco_names.txt");
        if (!class_file) {
            std::cerr << "failed to open classes.txt\n";
            return 0;
        }

        std::string line;
        while (std::getline(class_file, line))
            class_names.push_back(line);
    }
    // Initialize the parameters para alocação de memoria for object detection using YOLOV4
    // faço load dos ficheiros de configuração do método YOLOV4
    //auto net = cv::dnn::readNetFromDarknet("custom-yolov4-detector.cfg", "custom-yolov4-detector_best.weights");
    //auto net = cv::dnn::readNetFromDarknet("yolov4.cfg", "custom-yolov4-tiny-detector_best.weights");
    //cv::dnn::Net net = cv::dnn::readNetFromONNX("best.onnx");
    auto net = cv::dnn::readNetFromONNX("yolov5.onnx");

    cout << "here" << endl;
    // using GPU for image processing
    //net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    // using CPU for image processing
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    auto output_names = net.getUnconnectedOutLayersNames();
    cv::Mat blob;
    std::vector<cv::Mat> detections;
    std::vector<int> indices[NUM_CLASSES];
    std::vector<cv::Rect> boxes[NUM_CLASSES];
    std::vector<float> scores[NUM_CLASSES];

    // Creates 4-dimensional blob from image.
    cv::dnn::blobFromImage(im_1, blob, 0.00392, cv::Size(im_1.rows, im_1.cols), cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);
    net.forward(detections, output_names);
    // object detection using YOLOV4
    for (auto& output : detections) {
        const auto num_boxes = output.rows;
        for (int i = 0; i < num_boxes; i++) {
            //calculo das 5 predições para cada bounding box: x, y, w, h , confiança
            auto x = output.at<float>(i, 0) * im_1.cols;
            auto y = output.at<float>(i, 1) * im_1.rows;
            auto width = output.at<float>(i, 2) * im_1.cols;
            auto height = output.at<float>(i, 3) * im_1.rows;
            cv::Rect rect(x - width / 2, y - height / 2, width, height);

            for (int c = 0; c < NUM_CLASSES; c++) {
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence >= CONFIDENCE_THRESHOLD) {
                    boxes[c].push_back(rect);
                    scores[c].push_back(confidence);
                }
            }
        }
    }
    // Realiza a supressão não máxima das bounding boxes e das pontuações  de confiança correspondentes.
    // eliminação de bounding boxes repetidas que identificam o mesmo objecto.
    for (int c = 0; c < NUM_CLASSES; c++)
        cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

    // identificação dos objectos e correspondentes pontuações de confiança através de bounding boxes.
    for (int c = 0; c < NUM_CLASSES; c++) {
        for (size_t i = 0; i < indices[c].size(); ++i) {
            const auto color = colors[c % NUM_COLORS];

            auto idx = indices[c][i];
            const auto& rect = boxes[c][idx];
            cv::rectangle(im_1, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

            // coloco a identificação da classe do objeto contido na bounding box - pedestre ou garrafa por ex.
            std::ostringstream label_ss;
            label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
            auto label = label_ss.str();

            int baseline;
            auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            // defino o rectangulo que define o objeto detectado
            cv::rectangle(im_1, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
            // coloco a identificação da classe do objecto detectado.
            cv::putText(im_1, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
        }
    }
    cv::namedWindow("YOLOV5 detection", cv::WINDOW_NORMAL);
    cv::imshow("YOLOV5 detection", im_1);
    cv::waitKey(0);
    cv::imwrite("YOLOV5_res.jpg", im_1);

    return 0;
}