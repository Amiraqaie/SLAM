#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>

using namespace std;

int main()
{
    cv::Mat image;
    image = cv::imread("/home/amir/SLAM/review/ch5/images/color/1.png");

    cout << "image size is : " << image.size() << endl;
    cout << "image cols : " << image.cols << endl;
    cout << "image rows : " << image.rows << endl;
    cout << "image type : " << image.type() << endl;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    for (size_t y = 0; y < image.rows; y++)
    {
        uint8_t *row_ptr = image.ptr<uint8_t>(y);
        for (size_t x = 0; x < image.cols; x++)
        {
            uint8_t *data_ptr = &row_ptr[x * image.channels()];
            for (int c = 0; c != image.channels(); c++)
            {
                uint8_t data  = data_ptr[c];
            }
        }
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> duration = chrono::duration_cast<chrono::duration<double>> (t2 - t1);

    cout << "time used to loop over all pixels = " << duration.count() << "seconds" << endl;


    return 0;
}