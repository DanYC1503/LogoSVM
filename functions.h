#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <opencv2/ml.hpp>
#include <opencv2/videoio.hpp>


void detectLogoWithSlidingWindow(const cv::Mat &image, const cv::Ptr<cv::ml::SVM> &svm,const std::map<int, std::string> &labelToName, float scaleFactor,
                                 float windowFraction, float certaintyThreshold);

void mergeOverlappingBoxes(const std::vector<cv::Rect> &boxes, const std::vector<std::string> &labels,
                           std::vector<cv::Rect> &mergedBoxes, std::vector<std::string> &mergedLabels);

bool verifyDetection(const cv::Mat &image, const cv::Rect &roi, const cv::Ptr<cv::ml::SVM> &svm,
                     const std::map<int, std::string> &labelToName, const std::string &expectedLabel);

void imageFolder(const std::string &folderPath, const cv::Ptr<cv::ml::SVM> &svm, 
                 const std::map<int, std::string> &labelToName);

void nonMaxSuppression(const std::vector<cv::Rect> &boxes, const std::vector<float> &scores,
                       const std::vector<std::string> &labels, float iouThreshold,
                       std::vector<cv::Rect> &finalBoxes, std::vector<std::string> &finalLabels);

void filterNestedBoxes(std::vector<cv::Rect>& boxes, std::vector<float>& scores);

cv::Mat resizeImage(const cv::Mat &inputImage, const cv::Size &targetSize);


#endif 
