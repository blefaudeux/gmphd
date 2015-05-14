#include <cv.h>
#include <highgui.h>
#include <motion_filtering.h>

void draw(IplImage *image,
          const float *measurements,
          const float *filtered_state,
          const float *predicted_state) {

    cvDrawCircle(image,cvPoint(measurements[0],measurements[1]),2,cvScalar(0,0,255),2);
    cvDrawCircle(image,cvPoint(filtered_state[0],filtered_state[1]),2,cvScalar(0,255,0),2);

    // Show the predicted motion vector (?)
    cvDrawLine(image, cvPoint(filtered_state[0],filtered_state[1]), cvPoint(predicted_state[0],predicted_state[1]), cvScalar(255,0,0),2);
  }


int main() {

  // Deal with the OpenCV window..
  float angle = CV_PI/2-0.03;
  unsigned int width   = 800;
  unsigned int height  = 800;

  int n_targets = 5;

  // vector<vector<float[2]> > vec_poses;
  // vec_poses.resize(5);

  IplImage * image = cvCreateImage(cvSize(width,height),8,3);

  // Instanciate the motion filters
  float poses[3] = {0,0,0};

  MotionEstimation **motion_estimators;
  motion_estimators = new MotionEstimation*[n_targets];

  for (unsigned int i=0; i<n_targets; ++i) {
      motion_estimators[i] = new MotionEstimation(&poses[i], 1e3, 1.0, 0.05);
    }

  // Track the circling targets
  float measurements[6] = {0,0,0,0,0,0};
  float filtered_state[6], predicted_state[6], previous_state[6];

  for(;;angle += 0.01) {
      cvZero(image);

      // Create a new measurement for every target, and do the update
      for (unsigned int i=0; i< n_targets; ++i) {
          motion_estimators[i]->getLatestState(previous_state);

          // Propagate the previous state
          motion_estimators[i]->predict();

          // Get the predicted state :
          motion_estimators[i]->getPropagatedState(predicted_state);

          // Update the state with a new noisy measurement :
          measurements[0] = (width>>1)  + 300*cos(angle) + (rand()%2==1?-1:1)*(rand()%50);
          measurements[1] = (height>>1) + 300*sin(angle) + (rand()%2==1?-1:1)*(rand()%50);

          // Define a new 'speed' measurement
          measurements[3] = measurements[0] - previous_state[0];
          measurements[4] = measurements[1] - previous_state[1];

          motion_estimators[i]->update(measurements);

          // Get the filtered state :
          motion_estimators[i]->getLatestState(filtered_state);

          //vec_poses[i].push_back({filtered_state[0], filtered_state[1]});

          // Draw both the noisy input and the filtered state :
          draw(image, measurements, filtered_state, predicted_state);
        }

      // Show this stuff
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
  for (int i=0; i<n_targets; ++i) {
      delete motion_estimators[i];
    }

  delete []motion_estimators;

  return 1;
}
