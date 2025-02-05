#include "functions.h"
#include "training.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <bits/stl_numeric.h>
namespace fs = std::filesystem;


// Function to merge overlapping boxes with the same label
void mergeOverlappingBoxes(const std::vector<cv::Rect> &boxes, const std::vector<std::string> &labels,
                           std::vector<cv::Rect> &mergedBoxes, std::vector<std::string> &mergedLabels) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        bool merged = false;
        for (size_t j = 0; j < mergedBoxes.size(); ++j) {
            if (labels[i] == mergedLabels[j] && (boxes[i] & mergedBoxes[j]).area() > 0) {
                // Merge overlapping boxes with the same label
                mergedBoxes[j] = boxes[i] | mergedBoxes[j];
                merged = true;
                break;
            }
        }
        if (!merged) {
            mergedBoxes.push_back(boxes[i]);
            mergedLabels.push_back(labels[i]);
        }
    }
}

// Function to verify detection on a merged region
bool verifyDetection(const cv::Mat &image, const cv::Rect &roi, const cv::Ptr<cv::ml::SVM> &svm,
                     const std::map<int, std::string> &labelToName, const std::string &expectedLabel) {
    cv::Mat window = image(roi);

    // Resize the window to the required size for HOG extraction
    cv::Mat resizedWindow = resizeImage(window, cv::Size(32, 64)); // Adjust size as needed

    // Compute HOG features
    std::vector<float> hogFeatures = computeHOG(resizedWindow);

    // Convert HOG features to cv::Mat and ensure it's of type CV_32F
    cv::Mat hogMat = cv::Mat(hogFeatures).reshape(1, 1);
    hogMat.convertTo(hogMat, CV_32F);

    // Make prediction
    int predictedLabel = static_cast<int>(svm->predict(hogMat));
    if (labelToName.find(predictedLabel) != labelToName.end()) {
        return labelToName.at(predictedLabel) == expectedLabel;
    }
    return false;
}

void detectLogoWithSlidingWindow(const cv::Mat &image, const cv::Ptr<cv::ml::SVM> &svm,
                                 const std::map<int, std::string> &labelToName, float scaleFactor,
                                 float windowFraction , float certaintyThreshold ) {
    // Image size and initial positions for sliding window
    int stepSize = 5;  // Adjust the step size for the window (can be tuned)
    int width = image.cols;
    int height = image.rows;

    // Define window size as a fraction of the image size (e.g., 20% of the image)
    int windowWidth = static_cast<int>(width * windowFraction);
    int windowHeight = static_cast<int>(height * windowFraction);
    cv::Size windowSize(windowWidth, windowHeight);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<std::string> boxLabels;

    // Loop over the image at different scales
    for (float scale = 1.2f; scale < 2.0f; scale *= scaleFactor) {
        cv::Size scaledWindowSize = cv::Size(windowSize.width * scale, windowSize.height * scale);

        // Scan the image with a sliding window
        for (int y = 0; y < height - scaledWindowSize.height; y += stepSize) {
            for (int x = 0; x < width - scaledWindowSize.width; x += stepSize) {
                cv::Rect roi(x, y, scaledWindowSize.width, scaledWindowSize.height);
                cv::Mat window = image(roi);

                // Convert the window to grayscale
                cv::Mat grayWindow;
                if (window.channels() > 1) {
                    cv::cvtColor(window, grayWindow, cv::COLOR_BGR2GRAY);
                } else {
                    grayWindow = window;
                }

                // Skip if the region is mostly uniform
                cv::Scalar meanColor = cv::mean(window);
                if (cv::countNonZero(grayWindow != meanColor[0]) == 0) {
                    continue;
                }

                // Resize the window to the required size for HOG extraction
                cv::Mat resizedWindow = resizeImage(window, windowSize);

                // Compute HOG features
                std::vector<float> hogFeatures = computeHOG(resizedWindow);

                // Convert HOG features to cv::Mat and ensure it's of type CV_32F
                cv::Mat hogMat = cv::Mat(hogFeatures).reshape(1, 1);
                hogMat.convertTo(hogMat, CV_32F);

                // Make prediction and get confidence score
                float confidence = svm->predict(hogMat, cv::noArray(), cv::ml::StatModel::RAW_OUTPUT);
                confidence = 1.0f / (1.0f + exp(-confidence)); // Convert to probability

                // Check if the confidence is above the threshold
                if (confidence >= certaintyThreshold) {
                    int predictedLabel = static_cast<int>(svm->predict(hogMat));
                    if (labelToName.find(predictedLabel) != labelToName.end()) {
                        boxes.push_back(roi);
                        scores.push_back(confidence);
                        boxLabels.push_back(labelToName.at(predictedLabel));
                    }
                }
            }
        }
    }
    // Filter out nested boxes
    filterNestedBoxes(boxes, scores);
    // Merge overlapping boxes with the same label
    std::vector<cv::Rect> mergedBoxes;
    std::vector<std::string> mergedLabels;
    mergeOverlappingBoxes(boxes, boxLabels, mergedBoxes, mergedLabels);

    // Verify detection on merged regions
    std::vector<cv::Rect> finalBoxes;
    std::vector<std::string> finalLabels;
    for (size_t i = 0; i < mergedBoxes.size(); ++i) {
        if (verifyDetection(image, mergedBoxes[i], svm, labelToName, mergedLabels[i])) {
            finalBoxes.push_back(mergedBoxes[i]);
            finalLabels.push_back(mergedLabels[i]);
        }
    }

    // Draw the final bounding boxes and print predictions
    for (size_t i = 0; i < finalBoxes.size(); ++i) {
        // Draw the bounding box
        cv::rectangle(image, finalBoxes[i], cv::Scalar(0, 255, 0), 2);

        // Position the text slightly below the top-left corner of the box
        cv::Point textPos(finalBoxes[i].tl().x, finalBoxes[i].tl().y - 10);

        // Make sure the text is within the image boundaries
        if (textPos.y < 10) {
            textPos.y = finalBoxes[i].tl().y + 20;
        }

        // Draw the label text
        cv::putText(image, finalLabels[i], textPos, cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 255, 0), 2);
    }

    // Save the image with bounding boxes and labels
    static int image_number = 0;
    std::string outputPath = "detects/detected_logo" + std::to_string(image_number++) + ".jpg";
    cv::imwrite(outputPath, image);
}

cv::Mat resizeImage(const cv::Mat &inputImage, const cv::Size &targetSize) {
    cv::Mat resizedImage;
    cv::resize(inputImage, resizedImage, targetSize);
    return resizedImage;
}
void imageFolder(const std::string &folderPath, const cv::Ptr<cv::ml::SVM> &svm, 
                 const std::map<int, std::string> &labelToName) {
    // Iterate through all files in the folder
    for (const auto &entry : std::filesystem::directory_iterator(folderPath)) {
        const std::string &imagePath = entry.path().string();
        cv::Mat image = cv::imread(imagePath);

        if (image.empty()) {
            std::cerr << "Error reading image: " << imagePath << std::endl;
            continue;
        }

        // Check if the image width is greater than 1080 and scale it down if necessary
        if (image.cols > 1080) {
            // Scale the image by a factor of 3
            cv::resize(image, image, cv::Size(image.cols / 2, image.rows / 2));
        }

        // Call detectLogoWithSlidingWindow for each image, using a window fraction (e.g., 0.2f for 20%)
        detectLogoWithSlidingWindow(image, svm, labelToName, 1.2f, 0.2f, 0.80f);
    }
}


// Función para aplicar Non-Maximum Suppression (NMS)
void nonMaximumSuppression(const std::vector<cv::Rect>& boxes, 
                           const std::vector<float>& scores, 
                           float overlapThreshold, 
                           std::vector<cv::Rect>& finalBoxes, 
                           std::vector<float>& finalScores) {
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Ordenar por confianza descendente
    std::sort(indices.begin(), indices.end(), [&scores](int i, int j) {
        return scores[i] > scores[j];
    });

    std::vector<bool> suppress(boxes.size(), false);

    for (size_t i = 0; i < indices.size(); ++i) {
        if (suppress[indices[i]]) continue;
        finalBoxes.push_back(boxes[indices[i]]);
        finalScores.push_back(scores[indices[i]]);

        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (suppress[indices[j]]) continue;

            float intersectionArea = (boxes[indices[i]] & boxes[indices[j]]).area();
            float unionArea = boxes[indices[i]].area() + boxes[indices[j]].area() - intersectionArea;
            float iou = intersectionArea / unionArea;

            if (iou > overlapThreshold) {
                suppress[indices[j]] = true; // Suprimir la caja de menor confianza
            }
        }
    }
}
// Check if box A is inside box B
bool isBoxInside(const cv::Rect& A, const cv::Rect& B) {
    return A.x >= B.x && A.y >= B.y && A.x + A.width <= B.x + B.width && A.y + A.height <= B.y + B.height;
}

// Function to ensure no boxes are inside one another
void filterNestedBoxes(std::vector<cv::Rect>& boxes, std::vector<float>& scores) {
    std::vector<cv::Rect> filteredBoxes;
    std::vector<float> filteredScores;

    for (size_t i = 0; i < boxes.size(); ++i) {
        bool isNested = false;
        for (size_t j = 0; j < boxes.size(); ++j) {
            if (i != j && isBoxInside(boxes[i], boxes[j])) {
                isNested = true;
                break;
            }
        }
        if (!isNested) {
            filteredBoxes.push_back(boxes[i]);
            filteredScores.push_back(scores[i]);
        }
    }

    boxes = filteredBoxes;
    scores = filteredScores;
}
