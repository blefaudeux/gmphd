#include <cv.h>
#include <highgui.h>
#include "gmphd_filter.h"

using namespace std;

bool isTargetVisible( float probaDetection )
{
  int const maxRand = probaDetection * RAND_MAX;
  return rand() < maxRand;
}

void initTargetTracking( GMPHD & tracker )
{
  // Birth model (spawn)
  GaussianModel Birth(2);
  Birth.m_weight = 0.2f;
  Birth.m_mean(0,0) = 400.f;
  Birth.m_mean(1,0) = 400.f;
  Birth.m_mean(2,0) = 0.f;
  Birth.m_mean(3,0) = 0.f;
  Birth.m_cov = 400.f * MatrixXf::Identity(4,4);

  vector<GaussianModel> BirthModel;
  BirthModel.push_back(Birth);
  tracker.setBirthModel( BirthModel );

  // Dynamics (motion model)
  tracker.setDynamicsModel( 1.f, 10.f );

  // Detection model
  float const probDetection = 0.5f;
  float const measNoisePose = 2.f;
  float const measNoiseSpeed = 2.f;
  float const measBackground = 1.f;
  tracker.setObservationModel( probDetection, measNoisePose, measNoiseSpeed, measBackground);

  // Pruning parameters
  tracker.setPruningParameters( 0.2f, 2.f, 10);

  // Spawn (target apparition)
  vector<SpawningModel> spawnModel(1);
  tracker.setSpawnModel(spawnModel);

  // Survival over time
  tracker.setSurvivalProbability(0.95f);

  // Check initalization
  tracker.isInitialized();
}


int main() {

  // Deal with the OpenCV window..
  unsigned int width   = 800;
  unsigned int height  = 800;

  IplImage * image = cvCreateImage(cvSize(width,height),8,3);

  // Declare the target tracker and initialize the motion model
  int const n_targets = 5;
  GMPHD targetTracker( n_targets, 2, true );
  initTargetTracking( targetTracker );

  // Track the circling targets
  vector<float> targetEstimPosition, targetEstimSpeed, targetEstimWeight;
  vector<float> targetMeasPosition, targetMeasSpeed;
  vector< pair<float, float> > previousPoses(n_targets);

  float measurements[2];
  float const detectionProbability = 0.5f;

  for(float angle = CV_PI/2 - 0.03f;; angle += 0.01)
  {
    cvZero(image);

    // Create a new measurement vector
    targetMeasPosition.clear();
    targetMeasSpeed.clear();

    for (unsigned int i=0; i< n_targets; ++i)
    {
      // For each target, randomly visible or not
      if( isTargetVisible(detectionProbability) )
      {
        // Update the state with a new noisy measurement :
        measurements[0] = (width>>1)  + 300*cos(angle) + (rand()%2==1?-1:1)*(rand()%50);
        measurements[1] = (height>>1) + 300*sin(angle) + (rand()%2==1?-1:1)*(rand()%50);

        targetMeasPosition.push_back( measurements[0]);
        targetMeasPosition.push_back( measurements[1]);

        // Define a new 'speed' measurement
        targetMeasSpeed.push_back( measurements[0] - previousPoses[i].first);
        targetMeasSpeed.push_back( measurements[1] - previousPoses[i].second);

        previousPoses[i].first  = measurements[0];
        previousPoses[i].second = measurements[1];
      }
    }

    // Update the tracker
    targetTracker.setNewMeasurements( targetMeasPosition, targetMeasSpeed );

    // Get all the predicted targets
    targetTracker.propagate();
    targetTracker.getTrackedTargets( targetEstimPosition, targetEstimSpeed, targetEstimWeight, 0.2f );

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

