/*
 * Benjamin Lefaudeux (blefaudeux@github)
 * Very basic example of a gmphd usecase, tracking moving targets on a 2D plane
 * whose measurement is polluted by random noise (both in terms of position and
 * detection probability). Rendering is pretty much as basic as it gets,
 * but you get the point
 */

#include "gmphd_filter.h"
#include <cv.h>
#include <highgui.h>

using namespace std;

bool isTargetVisible(float probaDetection) {
  int const maxRand = probaDetection * RAND_MAX;
  return rand() < maxRand;
}

// Init the Gaussian Mixture Probability Hypothesis Density filter
void initTargetTracking(GMPHD &tracker) {
  // Birth model (spawn)
  GaussianModel Birth(4);
  Birth.m_weight = 0.2f;
  Birth.m_mean(0, 0) = 400.f;
  Birth.m_mean(1, 0) = 400.f;
  Birth.m_mean(2, 0) = 0.f;
  Birth.m_mean(3, 0) = 0.f;
  Birth.m_cov = 400.f * MatrixXf::Identity(4, 4);

  vector<GaussianModel> BirthModel;
  BirthModel.push_back(Birth);
  tracker.setBirthModel(BirthModel);

  // Dynamics (motion model)
  tracker.setDynamicsModel(1.f, 10.f);

  // Detection model
  float const probDetection = 0.5f;
  float const measNoisePose = 2.f;
  float const measNoiseSpeed = 20.f;
  float const measBackground = 0.5f;
  tracker.setObservationModel(probDetection, measNoisePose, measNoiseSpeed,
                              measBackground);

  // Pruning parameters
  tracker.setPruningParameters(0.3f, 3.f, 10);

  // Spawn (target apparition)
  vector<SpawningModel> spawnModel(1);
  tracker.setSpawnModel(spawnModel);

  // Survival over time
  tracker.setSurvivalProbability(0.95f);

  // (useless : check that the tracker is properly initialized)
  tracker.isInitialized();
}

bool display(vector<float> const &measures, vector<float> const &filtered,
             vector<float> const &weight, IplImage &pict) {
  // Display measurement hits
  for (auto tgt = measures.begin(); tgt != measures.end(); tgt++) {
    cvDrawCircle(&pict, cvPoint(*tgt, *(++tgt)), 2, cvScalar(0, 0, 255), 2);
  }

  // Display filter output
  float const scale = 5.f;
  auto w = weight.begin();
  for (auto tgt = filtered.begin(); tgt != filtered.end(); tgt++) {
    cvDrawCircle(&pict, cvPoint(*tgt, *(++tgt)), *(w++) * scale,
                 cvScalar(200, 0, 200), 2);
  }

  cvShowImage("Filtering results", &pict);

  printf("-----------------------------------------------------------------\n");
  int const k = cvWaitKey(100);
  return (k != 27) && (k != 1048603);
}

int main() {

  // Deal with the OpenCV window..
  unsigned int width = 800;
  unsigned int height = 800;

  IplImage *image = cvCreateImage(cvSize(width, height), 8, 3);

  // Declare the target tracker and initialize the motion model
  int const n_targets = 5;
  GMPHD targetTracker(n_targets, 2, true);
  initTargetTracking(targetTracker);

  // Track the circling targets
  vector<float> targetEstimPosition, targetEstimSpeed, targetEstimWeight;
  vector<float> targetMeasPosition, targetMeasSpeed;
  vector< pair<float, float> > previousPoses(n_targets);

  float measurements[2];
  float const detectionProbability = 0.5f;

  for (float angle = CV_PI / 2 - 0.03f;; angle += 0.01) {
    cvZero(image);

    targetMeasPosition.clear();
    targetMeasSpeed.clear();

    for (unsigned int i = 0; i < n_targets; ++i) {
      // For each target, randomly visible or not
      // Make up noisy measurements
      if (isTargetVisible(detectionProbability)) {
        measurements[0] = (width >> 1) + 300 * cos(angle) +
                          (rand() % 2 == 1 ? -1 : 1) * (rand() % 50);
        measurements[1] = (height >> 1) + 300 * sin(angle) +
                          (rand() % 2 == 1 ? -1 : 1) * (rand() % 50);

        targetMeasPosition.push_back(measurements[0]);
        targetMeasPosition.push_back(measurements[1]);

        targetMeasSpeed.push_back(
            0.f /* measurements[0] - previousPoses[i].first */);
        targetMeasSpeed.push_back(
            0.f /* measurements[1] - previousPoses[i].second */);

        previousPoses[i].first = measurements[0];
        previousPoses[i].second = measurements[1];
      }
    }

    // Update the tracker
    targetTracker.setNewMeasurements(targetMeasPosition, targetMeasSpeed);

    // Get all the predicted targets
    targetTracker.propagate();
    targetTracker.getTrackedTargets(targetEstimPosition, targetEstimSpeed,
                                    targetEstimWeight, 0.2f);

    // Show our drawing
    if (!display(targetMeasPosition, targetEstimPosition, targetEstimWeight,
                 *image)) {
      break;
    }
  }

  cvReleaseImage(&image);
  return 1;
}
