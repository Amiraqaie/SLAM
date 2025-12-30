// The heart of our implimentation - the main() function

#include <iostream>
#include <string>
#include <cstring>
#include <glog/logging.h>
// TODO : add includes libraries
// TODO : add includes full system
// TODO : add parsing functions

using namespace std;
// TODO : using namespace ldso;

// Flags
std::string vignette;
std::string gammaCalib;
std::string source;
std::string calib;
std::string output_file;
std::string vocPath;


double rescale = 1;
bool reverseplay = false;
bool disableROS = false;
int startIdx = 1;
int endIdx = 1000000;
bool prefetch = false;
float playbackSpeed = 0;
bool preloading = false;
bool useSampleOutput = false;


void parseArgument(char* arg) {
    std::cout << "Argument = " << arg << std::endl;
    int option;
    float foption;
    char buf[1000];

    if (sscanf(arg, "sampleoutput=%d", &option) == 1) {
        if (option == 1)
        {
            useSampleOutput = true;
            std::cout << "Using Sample Output" << std::endl;
        }
        else
        {
            useSampleOutput = false;
            std::cout << "Not Using Sample Output" << std::endl;
        }
    }

    if (sscanf(arg, "quit=%d", &option) == 1) {
        if (option == 1)
        {
            setting_debugout_runquiet = true;
            std::cout << "Quit Mode" << std::endl;
        }
        else
        {
            setting_debugout_runquiet = false;
            std::cout << "Verbose Mode" << std::endl;
        }
    }
     
}

int main(int argc, char** argv)
{
    std::cout << "Hello World" << std::endl;
    
    bool FLAGS_colorlogtostderr = true;

    // parsing arguments
    for (size_t i = 1; i < argc; i++)
    {
        std::cout << i << ". : Parsing Argument : < " << argv[i] << " >" << std::endl;
        parseArgument(argv[i]);
    }
    

    return 0;
}