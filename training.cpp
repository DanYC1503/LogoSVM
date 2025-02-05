#include "training.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <random>
#include <map>
#include <fstream>
#include <sstream>
#include <tinyxml2.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>


namespace fs = std::filesystem;

// Function to load XML files from the annotations folder
void loadXmlFiles(const std::string& logosPath, std::vector<std::string>& xmlFiles) {
    // Read XML files in the annotation folder (Logos_Info/Annotations)
    cv::glob("Logos_Info/Annotations/*.xml", xmlFiles, false);
    std::cout << "Found " << xmlFiles.size() << " XML files in Logos_Info/Annotations" << std::endl;
}

// Function to load an image from the logos folder based on the file path
bool loadImage(const std::string& imagePath, cv::Mat& image) {
    image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return false;
    }
    std::cout << "Loaded image: " << imagePath << std::endl;
    return true;
}


void loadDatasetFromAnnotations(const std::string& logosPath, cv::Mat& fullData, cv::Mat& fullLabels, const std::map<std::string, int>& labelMap) {
    // Get the list of XML files from the annotations folder
    std::vector<std::string> xmlFiles;
    loadXmlFiles(logosPath, xmlFiles);

    int imagesFound = 0; // Variable to track the number of found images

    for (const std::string& xmlFile : xmlFiles) {
        // Parse the XML file using TinyXML2
        tinyxml2::XMLDocument doc;
        if (doc.LoadFile(xmlFile.c_str()) != tinyxml2::XML_SUCCESS) {
            continue;
        }

        // Get the root element: <annotation>
        tinyxml2::XMLElement* root = doc.FirstChildElement("annotation");

        // Get image filename (the filename should be directly under the logosPath)
        const char* filename = root->FirstChildElement("filename")->GetText();
        std::string imagePath = logosPath + "/" + filename;  // Combine logosPath and filename

        // Load the image
        cv::Mat image;
        if (!loadImage(imagePath, image)) {
            continue;
        }

        // Increment imagesFound if the image is successfully loaded
        imagesFound++;

        // Loop through all objects in the XML file (supports multiple objects per image)
        for (tinyxml2::XMLElement* objectElement = root->FirstChildElement("object"); objectElement != nullptr; objectElement = objectElement->NextSiblingElement("object")) {
            const char* className = objectElement->FirstChildElement("name")->GetText();

            // Find the label corresponding to the className in the labelMap
            if (labelMap.find(className) != labelMap.end()) {
                int label = labelMap.at(className);

                // Get the bounding box for the object
                tinyxml2::XMLElement* bndbox = objectElement->FirstChildElement("bndbox");
                float xmin = std::stof(bndbox->FirstChildElement("xmin")->GetText());
                float ymin = std::stof(bndbox->FirstChildElement("ymin")->GetText());
                float xmax = std::stof(bndbox->FirstChildElement("xmax")->GetText());
                float ymax = std::stof(bndbox->FirstChildElement("ymax")->GetText());

                // Optionally, crop the image to the bounding box
                cv::Rect roi(xmin, ymin, xmax - xmin, ymax - ymin);
                cv::Mat croppedImage = image(roi);

                // Compute HOG descriptors for the cropped image
                std::vector<float> hogDescriptors = computeHOG(croppedImage);
                                
                // If HOG descriptors are valid, add them to fullData
                if (!hogDescriptors.empty()) {
                    cv::Mat descriptorMat(1, hogDescriptors.size(), CV_32F);
                    for (size_t i = 0; i < hogDescriptors.size(); i++) {
                        descriptorMat.at<float>(0, i) = hogDescriptors[i];
                    }
                    fullData.push_back(descriptorMat);

                    // Add label to fullLabels
                    fullLabels.push_back(label);

                    // Generate a save path for HOG visualization
                    std::string savePath = "scans/" + std::string(filename) + "_hog.png";

                    // Save and visualize the HOG descriptor for this cropped image
                    saveAndVisualizeHOG(croppedImage, savePath);
                }

            }
        }
    }

    // Print the total number of images found
    std::cout << "Found images: " << imagesFound << std::endl;
    
    // Print dataset information
    std::cout << "Loaded Dataset\n";
    std::cout << "Total Samples: " << fullData.rows << std::endl;
    std::cout << "Total Labels: " << fullLabels.rows << std::endl;
}



cv::Ptr<cv::ml::SVM> trainSVM(const cv::Mat &trainData, const cv::Mat &trainLabels,
                              const cv::Mat &testData, const cv::Mat &testLabels) {
    if (trainData.rows != trainLabels.rows || testData.rows != testLabels.rows) {
        std::cerr << "Mismatch between number of samples in data and labels!" << std::endl;
        return nullptr;
    }

    // Crear el modelo SVM
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setGamma(0.01);
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-6));

    // Convertir datos a float para el entrenamiento
    cv::Mat trainDataFloat, trainLabelsInt;
    trainData.convertTo(trainDataFloat, CV_32F);
    trainLabels.convertTo(trainLabelsInt, CV_32S);

    // **Asignar pesos a las clases manualmente**
    cv::Mat classWeights = cv::Mat::zeros(5, 1, CV_32F);  // Para 5 clases (1-5)
    classWeights.at<float>(0) = 2.01f;  // Pepsi (peso positivo)
    classWeights.at<float>(1) = 2.01f;  // Playstation (peso positivo)
    classWeights.at<float>(2) = 2.01f;  // Shell (peso positivo)
    classWeights.at<float>(3) = 2.01f;  // Seagate (peso positivo)
    classWeights.at<float>(4) = 2.01f;  // Skittles (peso positivo)

    //classWeights.at<float>(5) = 0.1f;   // No-logo (peso más bajo)

    // Establecer los pesos en el SVM
    svm->setClassWeights(classWeights);

    // Entrenar el modelo SVM
    svm->train(trainDataFloat, cv::ml::ROW_SAMPLE, trainLabelsInt);

    // Guardar el modelo entrenado
    svm->save("svm_model.xml");
    std::cout << "Saved SVM model";

    // **Evaluación con datos de prueba**
    cv::Mat testDataFloat, testLabelsInt;
    testData.convertTo(testDataFloat, CV_32F);
    testLabels.convertTo(testLabelsInt, CV_32S);

    int correctPredictions = 0;
    for (int i = 0; i < testData.rows; i++) {
        cv::Mat sample = testDataFloat.row(i);
        float predictedLabel = svm->predict(sample);
        if (predictedLabel == testLabelsInt.at<int>(i, 0)) {
            correctPredictions++;
        }
    }

    // Imprimir resultados de la evaluación
    float accuracy = (correctPredictions / static_cast<float>(testData.rows)) * 100.0f;
    std::cout << "Test Accuracy: " << accuracy << "% (" << correctPredictions << "/" << testData.rows << " correct)\n";

    return svm;
}


std::vector<float> computeHOG(const cv::Mat& image) {
    // Define HOG descriptor parameters for CPU computation
    cv::HOGDescriptor hog(
        cv::Size(32, 64),  // winSize: The size of the detection window. A smaller window makes training faster but may lose detail.
        cv::Size(16, 16),   // blockSize: The size of each block (group of cells). Larger blocks capture more context but reduce local details.
        cv::Size(8, 8),   // blockStride: The step size when sliding blocks across the image. A large stride speeds up processing but reduces overlap.
        cv::Size(8, 8),   // cellSize: The size of each cell inside a block. Larger cells speed up computation but reduce precision.
        6                   // nbins: The number of histogram bins for gradient directions. More bins improve precision but slow down processing.
    );


    std::vector<float> descriptors;

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(32, 64));

    cv::Mat grayImage;
    if (resizedImage.channels() == 3) {
        cv::cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = resizedImage.clone();
    }

    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(grayImage);

    cv::Mat processedImage;
    cv::resize(grayImage, processedImage, cv::Size(128, 256));

    hog.compute(processedImage, descriptors);

    float norm = std::sqrt(std::inner_product(descriptors.begin(), descriptors.end(), descriptors.begin(), 0.0f)) + 1e-7f;
    for (float& val : descriptors) {
        val = std::min(val / norm, 0.2f);
    }

    norm = std::sqrt(std::inner_product(descriptors.begin(), descriptors.end(), descriptors.begin(), 0.0f)) + 1e-7f;
    for (float& val : descriptors) {
        val /= norm;
    }

    return descriptors;
}

cv::Mat visualizeHOG(const cv::Mat& image, const std::vector<float>& descriptors) {
    int width = image.cols;
    int height = image.rows;
    int cellSize = 8;  
    int blockSize = 16; 

    int numCellsX = width / cellSize;
    int numCellsY = height / cellSize;

    cv::Mat hogImage = cv::Mat::zeros(height, width, CV_8UC3);  // Empty image for visualization

    int descriptorIndex = 0;
    for (int y = 0; y < numCellsY; ++y) {
        for (int x = 0; x < numCellsX; ++x) {
            // Each HOG cell has 9 bins (for gradient directions)
            for (int bin = 0; bin < 9; ++bin) {
                float magnitude = descriptors[descriptorIndex + bin];
                float angle = bin * 20.0f;  // 20 degrees per bin

                // Scale magnitude for better visualization
                magnitude = magnitude * 50.0f;  // Adjust scaling factor as needed

                // Calculate the center of the cell in the hogImage
                int cellX = x * cellSize + cellSize / 2;
                int cellY = y * cellSize + cellSize / 2;

                // Convert angle to radians
                float radian = angle * CV_PI / 180.0f;

                // Calculate the end point of the line (gradient vector)
                cv::Point direction(
                    cellX + static_cast<int>(magnitude * cos(radian)),
                    cellY - static_cast<int>(magnitude * sin(radian))
                );

                // Draw the line (gradient vector) for the cell
                cv::line(hogImage, cv::Point(cellX, cellY), direction, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            }
            descriptorIndex += 9; // Move to the next cell descriptor
        }
    }

    // Return the visualized HOG image
    return hogImage;
}
void saveAndVisualizeHOG(const cv::Mat& image, const std::string& savePath) {
    std::vector<float> hogDescriptors = computeHOG(image);
    if (hogDescriptors.empty()) {
        std::cerr << "HOG descriptors are empty for visualization." << std::endl;
        return;
    }

    // Visualization (HOG image)
    cv::Mat visualizedHOG = visualizeHOG(image, hogDescriptors);

    // Resize the visualized HOG image to 256x256
    cv::Mat resizedHOG;
    cv::resize(visualizedHOG, resizedHOG, cv::Size(256, 256));

    // Create the 'scans' folder if it doesn't exist
    fs::create_directory("scans");

    // Save the visualization as an image in the 'scans' folder
    cv::imwrite(savePath, resizedHOG);
}
void splitDataset(const cv::Mat &fullData, const cv::Mat &fullLabels, 
                  cv::Mat &trainData, cv::Mat &trainLabels, 
                  cv::Mat &testData, cv::Mat &testLabels, float trainRatio) {
    std::vector<int> indices(fullData.rows);
    std::iota(indices.begin(), indices.end(), 0);  // Fill with sequential indices
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    int trainSize = static_cast<int>(trainRatio * fullData.rows);

    // Split data into training set
    for (int i = 0; i < trainSize; i++) {
        trainData.push_back(fullData.row(indices[i]));
        trainLabels.push_back(fullLabels.row(indices[i]));
    }

    // Split data into test set
    for (int i = trainSize; i < fullData.rows; i++) {
        testData.push_back(fullData.row(indices[i]));
        testLabels.push_back(fullLabels.row(indices[i]));
    }

    // Print the sizes of the splits
    std::cout << "Training set size: " << trainData.rows << " samples\n";
    std::cout << "Test set size: " << testData.rows << " samples\n";

    // Print the size of row 0 in training data
    if (trainData.rows > 0) {
        std::cout << "Size of training data row 0: " << trainData.row(0).cols << " columns\n";
        std::cout << "Size of training label row 0: " << trainLabels.row(0).cols << " columns\n";
    }
}

void loadLabelMap(const std::string& labelMapPath, std::map<std::string, int>& labelMap) {
    std::ifstream file(labelMapPath);
    if (!file.is_open()) {
        std::cerr << "Error opening label map file: " << labelMapPath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string className;
        int label;
        
        // Parse the line, ensuring no leading/trailing spaces
        if (std::getline(iss, className, ':') && (iss >> label)) {
            // Trim any leading/trailing spaces from className
            className = trim(className);
            labelMap[className] = label;
        }
    }
    std::cout << "Label map loaded. Total classes: " << labelMap.size() << std::endl;
}

std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    size_t end = str.find_last_not_of(" \t\n\r");
    return (start == std::string::npos || end == std::string::npos) ? "" : str.substr(start, end - start + 1);
}