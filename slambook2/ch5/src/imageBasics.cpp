#include <iostream>
#include <chrono>

using namespace std;

#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  cv::Mat image;
  image = cv::imread(argv[1]); 

  if (image.data == nullptr) { 
    cerr << "file" << argv[1] << " not exist." << endl;
    return 0;
  }

  cout << "Image cols: " << image.cols <<  ", rows: " << image.rows << ", channels: " << image.channels() << endl;
  cv::imshow("image", image);      
  cv::waitKey(0);                  

  if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
    cout << "image type incorrect." << endl;
    return 0;
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for (size_t y = 0; y < image.rows; y++) {
    unsigned char *row_ptr = image.ptr<unsigned char>(y); 
    for (size_t x = 0; x < image.cols; x++) {
      unsigned char *data_ptr = &row_ptr[x * image.channels()]; 
      for (int c = 0; c != image.channels(); c++) {
        unsigned char data = data_ptr[c]; 
      }
    }
  }

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast < chrono::duration < double >> (t2 - t1);
  cout << "time used: " << time_used.count() << " seconds." << endl;

  cv::Mat image_another = image;
  image_another(cv::Rect(0, 0, 100, 100)).setTo(0); 
  cv::imshow("image", image);
  cv::waitKey(0);

  cv::Mat image_clone = image.clone();
  image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
  cv::imshow("image", image);
  cv::imshow("image_clone", image_clone);
  cv::waitKey(0);

  // draw a rectangle
  // Define rectangle coordinates (top-left and bottom-right points)
  cv::Point topLeft(50, 50);
  cv::Point bottomRight(200, 150);
  // Define rectangle color (BGR) and thickness
  cv::Scalar color(0, 255, 0); // Green color
  int thickness = 2;   
  // Draw the rectangle
  cv::rectangle(image, topLeft, bottomRight, color, thickness);


  cv::destroyAllWindows();
  return 0;
}