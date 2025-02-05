#ifndef TRAINING_H
#define TRAINING_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>

void loadDatasetFromAnnotations(const std::string& logosPath, cv::Mat& fullData, cv::Mat& fullLabels, const std::map<std::string, int>& labelMap);
void loadXmlFiles(const std::string& logosPath, std::vector<std::string>& xmlFiles, const std::vector<std::string>& imageList);
bool loadImage(const std::string& imagePath, cv::Mat& image) ;
// Function to train an SVM with the loaded data
cv::Ptr<cv::ml::SVM> trainSVM(const cv::Mat &trainData, const cv::Mat &trainLabels,
                              const cv::Mat &testData, const cv::Mat &testLabels);

// Function to compute the HOG descriptor of an image
std::vector<float> computeHOG(const cv::Mat& image);

std::vector<cv::Mat> augmentImage(const cv::Mat& image, const std::vector<float>& scales);
// Function prototype
void saveAndVisualizeHOG(const cv::Mat& image, const std::string& savePath);

void splitDataset(const cv::Mat &fullData, const cv::Mat &fullLabels, 
                  cv::Mat &trainData, cv::Mat &trainLabels, 
                  cv::Mat &testData, cv::Mat &testLabels, float trainRatio);

void loadLabelMap(const std::string& labelMapPath, std::map<std::string, int>& labelMap) ;
std::string trim(const std::string& str);
#endif // TRAINING_H
