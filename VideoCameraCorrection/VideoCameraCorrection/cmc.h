#ifndef CMC_H
#define CMC_H

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/videostab.hpp"


class CGMC
{
public:
    CGMC(int downscale=2, int nb_features=4000, bool detections_masking=false);

    cv::Mat apply(cv::Mat frame, bool &ok, std::vector<cv::Rect> detections={});

private:
    int m_downscale;
    int m_numberFeatures;
    bool m_detectionsMasking;

    cv::Ptr<cv::videostab::MotionEstimatorRansacL2> m_est;
    cv::Ptr<cv::videostab::KeypointBasedMotionEstimator> m_kbest;

    cv::Mat m_prevFrame;
    cv::Mat m_prevWarp;
};

#endif // CMC_H
