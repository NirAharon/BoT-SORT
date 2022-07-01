#include <string>
#include <iostream>
#include <sstream>
#include <iterator>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <experimental/filesystem>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/videostab.hpp"

// --------------------------------------------
// Global Parameters
// --------------------------------------------
std::string dataPath = "~/Datasets";
std::string savePath = "./Results";
int downscale = 2;

bool WITH_MASKING = false; // true | false
std::string detectionsPath = "";

// --------------------------------------------

int GMC(int dataNumber, int seqNumber, bool test=false);

int main()
{
    // --------------------------------------------
    // Parameters
    // --------------------------------------------
    int seqNumber = 2;
    int dataNumber = 17; // 17 | 20
    bool testSet = false;
    bool ALL_SEQENCES = false;
    // --------------------------------------------

    std::vector<int> mot17train{2, 4, 5, 9, 10, 11, 13};
    std::vector<int> mot17test{1, 3, 6, 7, 8, 12, 14};
    std::vector<int> mot20train{1, 2, 3, 5};
    std::vector<int> mot20test{4, 6, 7, 8};

    std::vector<std::vector<int>> seqences{mot17train, mot17test, mot20train, mot20test};
    std::vector<int> datasets{17, 17, 20, 20};

    if (ALL_SEQENCES)
    {
        for (int d = 0; d < (int)datasets.size(); ++d)
        {
            dataNumber = datasets[d];
            testSet = d % 2 == 1;

            for (int s = 0; s < (int)seqences[d].size(); ++s)
            {
                seqNumber = seqences[d][s];
                GMC(dataNumber, seqNumber, testSet);
            }
        }
    }
    else
    {
        return GMC(dataNumber, seqNumber, testSet);
    }

    return 0;
}

int GMC(int dataNumber, int seqNumber, bool test)
{
    if (dataNumber != 17 && dataNumber != 20)
    {
        std::cout << "Unkown dataset MOT" << dataNumber << std::endl;
        return -1;
    }

    std::string dataString = "MOT" + std::to_string(dataNumber);
    std::string seqString = (seqNumber < 10 ? "0" : "") + std::to_string(seqNumber);
    std::cout << "Start generating GMC for " << dataString << "-" << seqString << std::endl;

    std::string EXT_NAME = dataNumber == 17 ? "-FRCNN" : "";
    std::string split = test ? "test" : "train";
    std::string path = dataPath + "/" + dataString + "/" + split +"/" + dataString + "-" + seqString + EXT_NAME + "/img1";
    std::vector<std::string> images;
    for (const auto & entry : std::experimental::filesystem::directory_iterator(path))
        images.push_back(entry.path());

    std::sort(images.begin(), images.end());

    std::ifstream detFile;
    if (WITH_MASKING)
    {
        std::string fullDetectionsPath = detectionsPath + "/YOLOX-" + dataString + "-" + seqString + "-det.txt";
        detFile.open(fullDetectionsPath);

        if (!detFile.is_open())
        {
            std::cout << "ERROR: Unable to open detection file: " << fullDetectionsPath << std::endl;
            return -2;
        }
    }

    std::string detLine;
    int detFrameNumber = -1;
    cv::Rect detRect;
    float detScore;

    int numFrames = (int)images.size();
    cv::Mat fullFrame, frame, prevFrame;

    cv::Ptr<cv::videostab::MotionEstimatorRansacL2> est = cv::makePtr<cv::videostab::MotionEstimatorRansacL2>(cv::videostab::MM_SIMILARITY);
    cv::Ptr<cv::videostab::KeypointBasedMotionEstimator> kbest = cv::makePtr<cv::videostab::KeypointBasedMotionEstimator>(est);

    if (!std::experimental::filesystem::is_directory(savePath) || !std::experimental::filesystem::exists(savePath))
    {
        std::experimental::filesystem::create_directory(savePath);
    }

    std::ofstream outFile;
    std::string fullSavePath = savePath + "/GMC-" + dataString + "-" + seqString + ".txt";
    outFile.open (fullSavePath);

    double overallTime = 0.0;
    cv::TickMeter timer;

    for (int i = 0; i < numFrames; ++i)
    {
        fullFrame = cv::imread(images[i], 0);

        if (fullFrame.empty())
        {
            std::cout << "ERROR: Empty frame " << images[i] << std::endl;
            continue;
        }

        timer.reset();
        timer.start();

        cv::Size downSize(fullFrame.cols / downscale, fullFrame.rows / downscale);
        cv::resize(fullFrame, frame, downSize, cv::INTER_LINEAR);

        cv::Mat warp = cv::Mat::eye(3, 3, CV_32F);

        if (!prevFrame.empty())
        {
            if (WITH_MASKING)
            {
                cv::Mat mask(frame.size(), CV_8U);
                mask.setTo(255);

                // Mask detections
                if (detFrameNumber == i)
                {
                    if (detScore > 0.5)
                        mask(detRect) = 0;
                }

                while(std::getline(detFile, detLine))
                {
                    std::stringstream ss(detLine);
                    std::vector<std::string> tokens;

                    while( ss.good() )
                    {
                        std::string substr;
                        getline( ss, substr, ',' );
                        tokens.push_back(substr);
                    }

                    detFrameNumber = std::stoi(tokens[0]) - 1;
                    detRect.x = int(std::stof(tokens[1]) / downscale);
                    detRect.y = int(std::stof(tokens[2]) / downscale);
                    detRect.width = int(std::stof(tokens[3]) / downscale);
                    detRect.height = int(std::stof(tokens[4]) / downscale);
                    detScore = std::stof(tokens[6]);

                    detRect.x = std::max(0, detRect.x);
                    detRect.y = std::max(0, detRect.y);
                    detRect.width = std::min(detRect.width, frame.cols - detRect.x);
                    detRect.height = std::min(detRect.height, frame.rows - detRect.y);

                    if (detFrameNumber > i)
                        break;

                    if (detScore > 0.5)
                        mask(detRect) = 0;
                }
                if (i % 100 == 0)
                {
                    cv::namedWindow("mask", cv::WINDOW_NORMAL);
                    cv::imshow("mask", mask);
                    cv::waitKey(0);
                }

                kbest->setFrameMask(mask);
            }

            bool ok;
            warp = kbest->estimate(prevFrame, frame, &ok);


            if (!ok)
            {
                std::cout << "WARNING: Warp not ok" << std::endl;
            }

            warp.convertTo(warp, CV_32F);
            warp.at<float>(0, 2) *= downscale;
            warp.at<float>(1, 2) *= downscale;
        }

        // Store last frame
        frame.copyTo(prevFrame);

        timer.stop();
        overallTime += timer.getTimeMilli();

        // Write result to file
        std::string line = std::to_string(i) + "\t" +
                std::to_string(warp.at<float>(0, 0)) + "\t" +
                std::to_string(warp.at<float>(0, 1)) + "\t" +
                std::to_string(warp.at<float>(0, 2)) + "\t" +
                std::to_string(warp.at<float>(1, 0)) + "\t" +
                std::to_string(warp.at<float>(1, 1)) + "\t" +
                std::to_string(warp.at<float>(1, 2)) + "\t";

        std::cout << line << std::endl;
        outFile << line << std::endl;
    }

    outFile.close();

    std::cout << "GMC time [mSec]: " << overallTime / numFrames << std::endl;
    std::cout << "Saved GMC to " << fullSavePath << std::endl;

    return 0;
}

