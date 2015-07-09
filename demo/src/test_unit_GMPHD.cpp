#include <cv.h>
#include <highgui.h>
#include "gmphd_filter.h"

struct pos
{
    pos()
    {
        m_x = 0.f;
        m_y = 0.f;
    }

    float m_x;
    float m_y;
};


bool isTargetVisible( float probaDetection )
{
    int const maxRand = probaDetection * RAND_MAX;
    return rand() < maxRand;
}

int main() {

    // Deal with the OpenCV window..
    float angle = CV_PI/2-0.03;
    unsigned int width   = 800;
    unsigned int height  = 800;

    int n_targets = 5;

    IplImage * image = cvCreateImage(cvSize(width,height),8,3);

    GMPHD targetTracking(n_targets, 2);

    // Track the circling targets
    std::vector<float> targetEstimPosition, targetEstimSpeed, targetEstimWeight;
    std::vector<float> targetMeasPosition, targetMeasSpeed;

    std::vector<pos> previousPoses(n_targets);
    float measurements[4];

    bool targetVisible = false;

    for(;;angle += 0.01) {
        cvZero(image);

        // Create a new measurement vector
        targetMeasPosition.clear();
        targetMeasSpeed.clear();

        for (unsigned int i=0; i< n_targets; ++i)
        {
            // For each target, randomly visible or not
            targetVisible = isTargetVisible(0.5);

            if( targetVisible )
            {
                // Update the state with a new noisy measurement :
                measurements[0] = (width>>1)  + 300*cos(angle) + (rand()%2==1?-1:1)*(rand()%50);
                measurements[1] = (height>>1) + 300*sin(angle) + (rand()%2==1?-1:1)*(rand()%50);

                targetMeasPosition.push_back( measurements[0]);
                targetMeasPosition.push_back( measurements[1]);

                // Define a new 'speed' measurement
                targetMeasSpeed.push_back( measurements[0] - previousPoses[i].m_x);
                targetMeasSpeed.push_back( measurements[1] - previousPoses[i].m_y);

                previousPoses[i].m_x = measurements[0];
                previousPoses[i].m_y = measurements[1];
            }
        }

        // Update the tracker
        targetTracking.setNewMeasurements( targetMeasPosition, targetMeasSpeed );

        // Get all the predicted targets
        targetTracking.propagate();
        targetTracking.getTrackedTargets(0.2f, targetEstimPosition, targetEstimSpeed, targetEstimWeight );

        // Show our drawing
        for ( unsigned int i=0; i<targetMeasPosition.size(); i+=2)
        {
            cvDrawCircle(image,cvPoint(targetMeasPosition[i], targetMeasPosition[i+1]), 2, cvScalar(0,0,255),2);
        }

        for ( unsigned int i=0; i<targetEstimPosition.size(); i+=2)
        {
            cvDrawCircle(image,cvPoint(targetEstimPosition[i], targetEstimPosition[i+1]), 2, cvScalar(0,0,255),2);
        }

        cvShowImage("image",image);
        printf("-----------------------------------------------------------------\n");
        int k = cvWaitKey(20);

        if ((k == 27) || (k == 1048603))
            break;
        else if (k != -1)
            printf("Key pressed : %d\n", k);
    }

    // Close everything and leave
    cvReleaseImage(&image);
    return 1;
}

