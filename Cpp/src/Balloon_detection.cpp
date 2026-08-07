/* This is the V1 for detecting an object and using a kalman filter to predict its position. As of right now writing this comment, there is a compatability issue with
the code so this has been left like this to wait incase the issue is resolved */

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>

int main() {
    std::vector<cv::Rect> bounding_boxes;
    std::vector<float> confidences;
    std::deque<float> area_history;
    const int WINDOW_SIZE = 30;
    const int input_size = 640;
    


    //lataa mallin
    //here you put your model path as well
    cv::dnn::Net net = cv::dnn::readNetFromONNX("/path/to/your/model.onnx");
    if (net.empty()) {
        std::cerr << "Error: Could not load the ONNX model." << std::endl;
        return -1;
    }

    //avaa kameran
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera." << std::endl;
        return -1;
    }

    
    cv::Mat frame;
     //kalman filterin alustaminen
        cv::KalmanFilter kalman(4, 2, 0);
        kalman.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                                                     0, 1, 0, 1,
                                                     0, 0, 1, 0,
                                                     0, 0, 0, 1);

        kalman.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0,
                                                     0, 1, 0, 0);

        kalman.processNoiseCov = (cv::Mat_<float>(4, 4) << 1e-4, 0, 0, 0,
                                                     0, 1e-4, 0, 0,
                                                     0, 0, 1e-4, 0,
                                                     0, 0, 0, 1e-4);

        kalman.measurementNoiseCov = (cv::Mat_<float>(2, 2) << 1e-1, 0,
                                                         0, 1e-1);
                                                     
        kalman.statePre = (cv::Mat_<float>(4, 1) << 0, 0, 0, 0);
        kalman.statePost = (cv::Mat_<float>(4, 1) << 0, 0, 0, 0);


    while (true) {
        cap >> frame;
        
        if (frame.empty()) {
            std::cerr << "Could not read frame." << std::endl;
            break;
        }

                                             
        //tyhjentää edelliset bounding boxit ja confidence arvot jotta muisti ei mene tukkoon
        const float scale_x = frame.cols / (float)input_size;
        const float scale_y = frame.rows / (float)input_size;
        bounding_boxes.clear();
        confidences.clear();

        //
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net.setInput(blob);
        cv::Mat output = net.forward();

        //ottaa blobin coordinaatit ja laskee lopulta bounding boxit ja confidence arvot
        for (int i = 0; i < output.size[2]; ++i) {
            if (output.at<float>(0, 4, i) > 0.5) {
                float x_center = output.at<float>(0, 0, i) * scale_x;
                float y_center = output.at<float>(0, 1, i) * scale_y;
                float width = output.at<float>(0, 2, i) * scale_x;
                float height = output.at<float>(0, 3, i) * scale_y;
                float confidence = output.at<float>(0, 4, i);

                float left = x_center - width / 2.0f;
                float top = y_center - height / 2.0f;

                cv::Rect box(left, top, width, height);
                bounding_boxes.push_back(box);
                confidences.push_back(confidence);
            }
           
            
        }

        //ottaa parhaimman bounding boxin ja käyttää sitä laskemaan alueen ja keskiarvon
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(bounding_boxes, confidences, 0.5, 0.2, nms_result);

        if (nms_result.empty()) {
            std::cout << "No bounding boxes detected after NMS." << std::endl;
        } else {
    int best_idx = nms_result[0];
    float best_confidence = confidences[nms_result[0]];
    for (int idx : nms_result) {
        if (confidences[idx] > best_confidence) {
            best_confidence = confidences[idx];
            best_idx = idx;
        }
    }

    cv::Mat prediction = kalman.predict();
    cv::Mat center_point = (cv::Mat_<float>(2, 1) << bounding_boxes[best_idx].x + bounding_boxes[best_idx].width / 2.0f,
                                                       bounding_boxes[best_idx].y + bounding_boxes[best_idx].height / 2.0f);
    cv::Mat estimated = kalman.correct(center_point);

    float area = bounding_boxes[best_idx].area();
    std::cout << "Bounding box area: " << area << " (confidence: " << best_confidence << ")" << std::endl;

    area_history.push_back(area);
    if (area_history.size() > WINDOW_SIZE) {
        area_history.pop_front();
    }

    float sum = 0.0f;
    for (float a : area_history) {
        sum += a;
    }
    float average_area = sum / area_history.size();
    std::cout << "Average area: " << average_area << std::endl;

    float est_x  = estimated.at<float>(0, 0);
    float est_y  = estimated.at<float>(1, 0);
    float est_vx = estimated.at<float>(2, 0);
    float est_vy = estimated.at<float>(3, 0);
    float N = 20.0f;
    float future_x = est_x + est_vx * N;
    float future_y = est_y + est_vy * N;

    cv::rectangle(frame, bounding_boxes[best_idx], cv::Scalar(0, 255, 0), 2);
    cv::circle(frame, cv::Point(future_x, future_y), 10, cv::Scalar(255, 0, 0), 2);
}

        cv::imshow("Balloon Detection", frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}