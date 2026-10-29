#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>

int main() {
    // 1. Capture an image using the lightning-fast native libcamera command
    std::string filename = "captured_image.jpg";
    std::string command = "rpicam-still -o " + filename + " --width 1280 --height 720 --nopreview -t 1000";
    
    std::cout << "Capturing image with Pi camera..." << std::endl;
    int result = std::system(command.c_str());
    
    if (result != 0) {
        std::cerr << "Error: Failed to capture image using rpicam-still." << std::endl;
        return -1;
    }
    
    // 2. Load the captured image into OpenCV for your image processing pipeline
    cv::Mat frame = cv::imread(filename);
    
    if (frame.empty()) {
        std::cerr << "Error: Could not read the saved image into OpenCV." << std::endl;
        return -1;
    }

    std::cout << "Image successfully loaded into OpenCV! Resolution: " 
              << frame.cols << "x" << frame.rows << std::endl;

    // TODO: Perform your OpenCV image processing here
    // Example: convert to grayscale
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    cv::imwrite("grayscale_image.jpg", grayFrame);
    std::cout << "Processed and saved grayscale_image.jpg successfully." << std::endl;

    return 0;
}
