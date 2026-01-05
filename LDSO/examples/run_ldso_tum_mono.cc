// The heart of our implimentation - the main() function

#include <iostream>
#include <string>
#include <cstring>
#include <glog/logging.h>
#include <thread>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/time.h>
#include <cmath>
#include <memory>
// #include "frontend/FullSystem.h"
// #include "DatasetReader.h"

// TODO : add includes libraries
// TODO : add includes full system
// TODO : add parsing functions

using namespace std;
// TODO : using namespace ldso;

// Flags
string vignette;
string gammaCalib;
string source;
string calib;
string output_file;
string vocPath;


double rescale = 1;
bool reverseplay = false;
bool disableROS = false;
int startIdx = 1;
int endIdx = 1000000;
bool prefetch = false;
float playbackSpeed = 0;
bool preload = false;
bool useSampleOutput = false;

// select preset settings (number of features, etc.).
void settingsDefault(int preset) {
    cout << "============== Selecting Preset Settings : " << preset << " ==============" << endl;
    if (preset == 0 || preset == 1) {
        cout << "Using Default Settings" << endl << (preset == 0 ? "no" : "1x")
        << " real - time enforcing " << endl
        << " 2000 active features" << endl
        << " 5 to 7 active frames " << endl
        << " 1 - 6 LM iterations per KF" << endl
        << " original image size" << endl;
        playbackSpeed = (preset == 0 ? 0 : 1);
        preload  = preset == 1;
        // TODO : setting_desiredImmatureDensity = 1500;
        // TODO : setting_desirePointDensity = 2000;
        // TODO : setting_minFrames = 5;
        // TODO : setting_maxFrames = 7;
        // TODO : setting_maxOptIterations = 6;
        // TODO : setting_minOptIterations = 1;
        // TODO : setting_logStuff = false;
    }

    if (preset == 2 || preset == 3) {
        cout << "Using FAST Settings" << endl << (preset == 2 ? "no" : "5x") 
        << " real - time enforcing " << endl
        << " 800 active features" << endl
        << " 4 to 6 active frames " << endl
        << " 1 - 4 LM iterations per KF" << endl
        << " 424 * 320 image size" << endl;
        playbackSpeed = (preset == 0 ? 0 : 5);
        preload  = preset == 3;
        // TODO : setting_desiredImmatureDensity = 600;
        // TODO : setting_desirePointDensity = 800;
        // TODO : setting_minFrames = 4;
        // TODO : setting_maxFrames = 6;
        // TODO : setting_maxOptIterations = 4;
        // TODO : setting_minOptIterations = 1;
        // TODO : benchmarkSetting_width = 424;
        // TODO : benchmarkSetting_hight = 320;
        // TODO : setting_logStuff = false;
    }
    cout << "============== Selecting Preset Settings : " << preset << " ==============" << endl;

}

void parseArgument(char* arg) {
    int option;
    float foption;
    char buff[1000];

    if (sscanf(arg, "sampleoutput=%d", &option) == 1) {
        if (option == 1)
        {
            useSampleOutput = true;
            cout << "Using Sample Output" << endl;
        }
        else
        {
            useSampleOutput = false;
            cout << "Not Using Sample Output" << endl;
        }
        return;
    }

    if (sscanf(arg, "quit=%d", &option) == 1) {
        if (option == 1)
        {
            // TODO : setting_debugout_runquiet = true;
            cout << "Quit Mode" << endl;
        }
        else
        {
            // TODO : setting_debugout_runquiet = false;
            cout << "Verbose Mode" << endl;
        }
        return;
    }

    if (sscanf(arg, "preset=%d", &option) == 1) {
        settingsDefault(option);
        return;
    }

    if (sscanf(arg, "rec=%d", &option) == 1) {
        if (option == 1)
        {
            // TODO : disableReconfigure = true;
            cout << "Disable RECONFIGURE" << endl;
        }
        else
        {
            // TODO : disableReconfigure = false;
            cout << "Enable RECONFIGURE" << endl;
        }
        return;
    }
    
    if (sscanf(arg, "noros=%d", &option) == 1) {
        if (option == 1)
        {
            // TODO : disableReconfigure = true;
            // TODO : disableROS = true;
            cout << "Disable ROS and RECONFIGURE" << endl;
        }
        else
        {
            // TODO : disableReconfigure = false;
            // TODO : disableROS = false;
            cout << "Enable ROS and RECONFIGURE" << endl;
        }
        return;
    }

    if (sscanf(arg, "nolog=%d", &option) == 1) {
        if (option == 1)
        {
            // TODO : setting_logStuff = false;
            cout << "Disable LOGGING" << endl;
        }
        else
        {
            // TODO : setting_logStuff = true;
            cout << "Enable LOGGING" << endl;
        }
        return;
    }

    if (sscanf(arg, "reversePlay=%d", &option) == 1) {
        if (option == 1)
        {
            reverseplay = true;
            cout << "Reverse Play" << endl;
        }
        else
        {
            reverseplay = false;
            cout << "Normal Play" << endl;
        }
        return;
    }

    if (sscanf(arg, "nogui=%d", &option) == 1) {
        if (option == 1)
        {
            // TODO : disableAllDisplay = true;
            cout << "Disable GUI" << endl;
        }
        else
        {
            // TODO : disableAllDisplay = false;
            cout << "Enable GUI" << endl;
        }
        return;
    }

    if (sscanf(arg, "nomt=%d", &option) == 1) {
        if (option == 1)
        {
            // TODO : multiThreading = false;
            cout << "No Multithreading" << endl;
        }
        else
        {
            // TODO : multiThreading = true;
            cout << "Multithreading" << endl;
        }
        return;
    }

    if (sscanf(arg, "prefetch=%d", &option) == 1) {
        if (option == 1)
        {
            prefetch = true;
            cout << "Prefetch" << endl;
        }
        else
        {
            prefetch = false;
            cout << "No Prefetch" << endl;
        }
        return;
    }

    if (sscanf(arg, "start=%d", &option) == 1) {
        startIdx = option;
        cout << "Start Index = " << startIdx << endl;
        return;
    }

    if (sscanf(arg, "end=%d", &option) == 1) {
        endIdx = option;
        cout << "End Index = " << endIdx << endl;
        return;
    }

    if (sscanf(arg, "loopclosing=%d", &option) == 1) {
        if (option == 1)
        {
            // TODO : setting_enableLoopClosing = true;
            cout << "Enable Loop Closing" << endl;
        }
        else
        {
            // TODO : setting_enableLoopClosing = false;
            cout << "Disable Loop Closing" << endl;
        }
        return;
    }

    if (sscanf(arg, "files=%s", buff) == 1) {
        source = buff;
        cout << "loading data from " << source << endl;
        return;
    }

    if (sscanf(arg, "vocab=%s", buff) == 1) {
        vocPath = buff;
        cout << "loading vocabulary from " << vocPath << endl;
        return;
    }

    if (sscanf(arg, "calib=%s", buff) == 1) {
        calib = buff;
        cout << "loading calibration from " << calib << endl;
        return;
    }

    if (sscanf(arg, "vignette=%s", buff) == 1) {
        vignette = buff;
        cout << "loading vignette from " << vignette << endl;
        return;
    }

    if (sscanf(arg, "gamma=%s", buff) == 1) {
        gammaCalib = buff;
        cout << "loading gammaCalib from " << gammaCalib << endl;
        return;
    }

    if (sscanf(arg, "rescale=%f", &foption) == 1) {
        playbackSpeed = foption;
        cout << "PLAYBACK SPEED " << playbackSpeed << endl;
        return;
    }

    if (sscanf(arg, "output=%s", buff) == 1) {
        output_file = buff;
        LOG(INFO) << "output file path" << output_file << endl;
        return;
    }

    if (sscanf(arg, "save=%d", &option) == 1) {
        if (option == 1)
        {
            // TODO : debugSaveImages = true;
            cout << "Enable Saving IMAGES!" << endl;

            // TODO : Do We really need this four times?
            if (42 == system("rm -rf images_out"))
            {
                LOG(ERROR) << "Could not remove images_out folder!" << endl;
            }
            if (42 == system("mkdir images_out"))
            {
                LOG(ERROR) << "Could not remove images_out folder!" << endl;
            }
            if (42 == system("rm -rf images_out"))
            {
                LOG(ERROR) << "Could not remove images_out folder!" << endl;
            }
            if (42 == system("mkdir images_out"))
            {
                LOG(ERROR) << "Could not remove images_out folder!" << endl;
            }
        }
        else
        {
            // TODO : debugSaveImages = false;
            cout << "Disable Saving Trajectory" << endl;
        }
        return;
    }

    if (sscanf(arg, "mode=%d", &option) == 1) {
        if (option == 0)
        {
            cout << "PHOTOMETRIC MODE WITH CALIBRATION" << endl;
        }
        else if (option == 1)
        {
            // TODO : setting_photometricCalibration = 0;
            // TODO : setting_affineOptModeA = 0; // -1: fix. >=0: optimize (with prior, if > 0)
            // TODO : setting_affineOptModeB = 0; // -1: fix. >=0: optimize (with prior, if > 0)
            cout << "PHOTOMETRIC MODE WITHOUT CALIBRATION" << endl;
        }
        else if (option == 2)
        {
            // TODO : setting_photometricCalibration = 1;
            // TODO : setting_affineOptModeA = -1; // -1: fix. >=0: optimize (with prior, if > 0)
            // TODO : setting_affineOptModeB = -1; // -1: fix. >=0: optimize (with prior, if > 0)
            // TODO : setting_minGradHistAdd = 3;
            cout << "PHOTOMETRIC MODE WITH PERFECT IMAGES" << endl;
        }
        return;
    }
    cout << "WARNING: Unknown argument " << arg << " ignored." << endl;
}

int main(int argc, char** argv)
{    
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;        // log to stderr instead of files
    FLAGS_colorlogtostderr = 1;  // colored output
    FLAGS_minloglevel = 0;       // INFO and above

    // parsing arguments
    for (int i = 1; i < argc; i++)
    {
        parseArgument(argv[i]);
    }

    // TODO : check setting conflicts
    /*
    if (setting_enableLoopCloosing && (setting_pointSelection != 1))
    {
        LOG(ERROR) << "Loop closing only works with 'HARRIS' point selection. Please set 'pointselection=1'." << endl;
        return -1;
    }
    */
   
    // TODO ; create and run system
    /*
    if (setting_showLoopClosing == true)
    {
        LOG(WARNING) << "Loop closing visualization requires GUI. Forcing GUI on. the program will be paused when loop is found " << endl;
    }
    */
    
    /* TODO : create data reader
    shared_ptr<ImageFolderReader> reader(new ImageFolderReader(ImageFolderReader::TUM_MONO,
    source,
    calib,
    gammaCalib,
    vignette));
    */

    // TODO : reader->setGlobalCalibration();

    /* TODO : check photometric calibration availability
    if (setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
    {
        LOG(ERROR) << "ERROR :  Dont have photometric calibration. Need to use commandline options mode=1 or mode=2" << endl;
        return -1;
    }
    */

    int lstart = startIdx;;
    int lend = endIdx;
    int linc = 1;

    if (reverseplay)
    {
        LOG(INFO) << "REVERSE PLAY ENABLED!" << endl;
        lstart = endIdx - 1;
        /* TODO : check end index
        if (lstart >= reader.getNumImages())
        {
            lstart = reader.getNumImages() - 1;
        }
        */
        lend = startIdx;
        linc = -1;
    }

    // Load the ORB Vocabulary for Loop Closing
    /*  TODO : load vocabulary
    shared_ptr<ORBVocabulary> voc(new ORBVocabulary());
    voc->load(vocPath);
    LOG(INFO) << "Loading ORB Vocabulary from " << vocPath << endl;
    */

    /* TODO : create full system
    shared_ptr<FullSystem> fullSystem(new FullSystem(voc));
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
    fullSystem->linearizeOperation = (playbackSpeed == 0);
    */

    /* TODO : create data feeder
    shared_ptr<PangolinDSOViewer> viewer = nullptr;
    if (!disableAllDisplay)
    {
        viewer = shared_ptr<PangolinDSOViewer>(new PangolinDSOViewer(wG[0], hG[0], false));
        fullSystem->setViewer(viewer);
    } else 
    {
        LOG(INFO) << "VIEWER DISABLED" << endl;
    }
    */

    // this is the main loop which runs on a separate thread
    std::thread runthread([&]() {

        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;
        

        /*  TODO : create data feeder prepare correct timestamps for each image
        for (int i = lstart; i >= 0 && i < reader->getNumImages() && linc * i < linc * lend; i += linc)
        {
            idsToPlay.push_back(i);
            if (timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back(0);
            }
            else
            {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
                timesToPlayAt.push_back(timesToPlayAt.back() + fabs(tsThis - tsPrev) / playbackSpeed);
            }
        }
        */
        
        /*  TODO : preload images if needed
        std::vector<ImageAndExposure> preloadedImages;
        if (preload)
        {
            LOG(INFO) << "PRELOADING IMAGES INTO MEMORY!" << endl;
            for (size_t ii = 0; ii < idsToPlay.size(); ii++)
            {
                int i = idsToPlay[ii];
                preloadedImages.push_back(reader->getImage(i));
            }
        }
        */

        // Start a stopwatch to measure performance of the run
        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset = 0;
        
        // Here is the main loop of the VSLAM system
        for (int ii = 0; ii < (int) idsToPlay.size(); ii++)
        {
            /* TODO : 
            while (setting_pause == true)
            {
                usleep(5000);
            }
            */

            /* TODO : 
            if (!fullSystem->initialized)
            {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }
            */

            int i = idsToPlay[ii];
            
            /* TODO : 
            ImageAndExposure *image; // The main data type for image
            if (preload)
            {
                img = preloadedImages[ii];
            }
            else
            {
                img = reader->getImage(i);
            }
            */

            bool skipFrame = false;
            if(playbackSpeed != 0)
            {
                struct timeval tv_now;
                gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) + (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));
                if (sSinceStart < timesToPlayAt[ii])
                {
                    // usleep((int) ((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));

                }
                else if (sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2))
                {
                    cout << "SKIPFRAME " << ii << "(play at " << timesToPlayAt[ii] << " ,now it is " << sSinceStart << ")!" << endl;
                skipFrame = true;
                }   
            }


            if (!skipFrame)
            {
                // here the "magic" is happening
                /* TODO : 
                fullSystem->addActiveFrame(img, i);
                */
            }
            /* TODO : 
            delete img;
            */
            
            // cleaning up
            /* TODO : 
            if (fullSystem->initFailed || setting_fullResetRequested)
            {
                if (ii < 250 || setting_fullResetRequested)
                {
                    LOG(INFO) << "Init Failed, RESETTING!";
                    fullSystem = shared_ptr<FullSystem>(new FullSystem(voc));
                    fullSystem->setGammaFunction(reader->getPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed == 0);
                    if (viewer)
                    {
                        viewer->reset();
                        sleep(1);
                        fullSystem->setViewer(viewer);
                    }
                    setting_fullResetRequire = false;
                }
            }
            */

            /* TODO : 
            if (fullSystem->isLost)
            {
                LOG(INFO) << "LOST!!!";
                break;
            }
            */

        }

        // TODO : fullSystem->blockUntilMappingFinished();

        // Stop the stopwatch
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);
        

        // helpfull logs
        // TODO : fullSystem->printResult(output_file, true);  // true = save loop closing results too
        // TODO : fullSystem->printResult(output_file + ".noloop", false);  // false = do not save loop closing results

        // TODO : (Segmentation fault)
        int numFramesPrecessed = 100; // int numFramesPrecessed = abs(idsToPlay[0] - idsToPlay.back());
        double numSecondsProcessed = 1234.0; // TODO : fabs(reader->getTimestamp(idsToPlay[0]) - reader->getTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f * (ended - started) / (float) (CLOCKS_PER_SEC);

        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
        
        cout 
        << "======== Performance Summary ========" 
        << endl
        << endl
        << endl
        << endl
        << endl
        << endl
        << endl;

        LOG(INFO) << "Processed " << numFramesPrecessed << " frames in "
                  << MilliSecondsTakenMT / 1000.0f << " s ("
                  << 1000.0f * numFramesPrecessed / MilliSecondsTakenMT
                  << " fps) , covering " << numSecondsProcessed << " s of video ("
                  << 100.0f * numSecondsProcessed / (MilliSecondsTakenMT / 1000.0f)
                  << " x real-time speed)." << endl;

        cout 
        << endl
        << endl
        << endl
        << endl
        << endl
        << "======== Performance Summary ========" 
        << endl;
        
        /* TODO : 
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f * (ended - started) / (float) (CLOCKS_PER_SEC) * reader->getNumImages() << " " 
            << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) / (float) reader->getNumImages() 
            << std::endl;
            
            tmlog.flush();
            tmlog.close();
        }
        */
    });
    

    /*
    if (viewer)
    {
        viewer->run();  // mac os should keep this in the main thread
    }
    viewer->saveAsPLYFille("pointcloud.ply");
    */
    
    runthread.join();  // this will wait until the processing thread is done
    LOG(INFO) << "Saved pointcloud to pointcloud.ply" << endl;
    LOG(INFO) << "Finished!" << endl;

    return 0;
}