#include "functions.h"
#include "training.h"
#include <opencv2/opencv.hpp>
#include <map>
#include <fstream>
#include <sstream>
#include <opencv2/core/cuda.hpp>


int main() {
    
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "CUDA is enabled and available!" << std::endl;
    } else {
        std::cout << "CUDA is not available." << std::endl;
    }
    std::string logosImages = "Logos";  // Path to images
    std::string logosXMLInfo = "Logos_Info";  // Path to XML annotations
    std::string labelMapFile = logosXMLInfo + "/labelmap.txt"; // Path to labelMap.txt
    std::string negativeImages = "images";
    // Containers for training data and labels
    cv::Mat fullData, fullLabels;

    // Load labelmap.txt to map class names to labels
    std::map<std::string, int> labelMap;
    loadLabelMap(labelMapFile, labelMap);
    // Load dataset from the new folder structure
    loadDatasetFromAnnotations(logosImages, fullData, fullLabels, labelMap);


    // Split data into train and test sets
    cv::Mat trainData, trainLabels, testData, testLabels;
    splitDataset(fullData, fullLabels, trainData, trainLabels, testData, testLabels, 0.8f);

    // Train the SVM model
    cv::Ptr<cv::ml::SVM> svm = trainSVM(trainData, trainLabels, testData, testLabels);
    std::cout << "Trained SVM_Model";
    std::string newImageDetection = "test";

    std::map<int, std::string> labelToName = {
    {1, "Pepsi"},
    {2, "Playstation"},
    {3, "Seagate"},
    {4, "Shell"},
    {5, "Skittles"},
    //{0, "Desconocido"} // Etiqueta para las imágenes negativas
    };

    imageFolder(newImageDetection, svm, labelToName);
}
