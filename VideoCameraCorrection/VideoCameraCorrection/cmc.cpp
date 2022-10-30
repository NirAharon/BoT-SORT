#include "cmc.h"


CGMC::CGMC(int downscale, int nb_features, bool detections_masking)
{
    if (0 < downscale && downscale < 8)
        m_downscale = downscale;

    if (0 < nb_features)
        m_numberFeatures = nb_features;

    m_detectionsMasking = detections_masking;

    m_est = cv::makePtr<cv::videostab::MotionEstimatorRansacL2>(cv::videostab::MM_SIMILARITY);
    m_kbest = cv::makePtr<cv::videostab::KeypointBasedMotionEstimator>(m_est);

    //kbest->setDetector(cv::FastFeatureDetector::create());
    //kbest->setDetector(cv::SiftFeatureDetector::create());
    m_kbest->setDetector(cv::GFTTDetector::create(4000));
}

cv::Mat CGMC::apply(cv::Mat fullFrame, bool &ok, std::vector<cv::Rect> detections)
{
    cv::Mat warp = cv::Mat::eye(3, 3, CV_32F);

    if (fullFrame.empty())
    {
        return cv::Mat();
    }

    // Downscale frame
    cv::Size downSize(fullFrame.cols / m_downscale, fullFrame.rows / m_downscale);
    cv::Mat frame;
    cv::resize(fullFrame, frame, downSize, cv::INTER_LINEAR);

    if (!m_prevFrame.empty())
    {
        if (m_detectionsMasking)
        {
            cv::Mat mask(frame.size(), CV_8U);
            mask.setTo(255);

            for (int d = 0; d < (int)detections.size(); ++d)
            {
                cv::Rect detRect = detections[d];
                detRect.x = std::max(0, detRect.x);
                detRect.y = std::max(0, detRect.y);
                detRect.width = std::min(detRect.width, frame.cols - detRect.x);
                detRect.height = std::min(detRect.height, frame.rows - detRect.y);

                mask(detRect) = 0;
            }

            m_kbest->setFrameMask(mask);
        }

        warp = m_kbest->estimate(m_prevFrame, frame, &ok);

        if (ok)
        {
            warp.convertTo(warp, CV_32F);
            warp.at<float>(0, 2) *= m_downscale;
            warp.at<float>(1, 2) *= m_downscale;
        }
        else
        {
            //std::cout << "WARNING: Warp not ok, using previous motion" << std::endl;
            m_prevWarp.copyTo(warp);
        }
    }

    // Store last frame
    frame.copyTo(m_prevFrame);
    warp.copyTo(m_prevWarp);


    return warp;
}
