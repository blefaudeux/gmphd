#include "motion_filtering.h"

/*
 *  @license GPL
 *  @author Benjamin Lefaudeux (blefaudeux at github)
 *  @file motion_filtering.cpp
 *  @brief Defines an instance of UKF filtering
 *  @version 1.0
 *  @date 12-11-2013
 */


/*!
 * \brief pointPropagation : propagation function for sigma points
 *  we propagate linear coordinates knowing speed, no angular values for now
 *
 * \param points_in
 * \param points_out
 */
void pointPropagation_speed(const MatrixXf &points_in, MatrixXf &points_out) {
  // Constant acceleration model... :
  MatrixXf prop_matrix;
  int dim = points_in.rows ();
  prop_matrix.setIdentity(dim,dim);
  prop_matrix(0,3) = 1.f;
  prop_matrix(1,4) = 1.f;
  prop_matrix(2,5) = 1.f;

  points_out.resizeLike(points_in);
  points_out = prop_matrix * points_in;
}

void pointPropagation_angularSpeed(const Quaternionf &q_in, Quaternionf &q_out) {
  // Constant velocity model up to now...
  // a bit lame..
  q_out = q_in;

  /*
  // Rotate the positions and speeds from the known iterative rotation :
  Matrix3f rot_mat;
  Vector3f rot_vec;
  Vector3f temp_vec;
  i = 0;

  while (i < sigma_points.size()) {
    rot_vec = sigma_points[i].block<3,1>(6,0) * this->time_step; // Angular offset

    rot_mat = AngleAxisf(rot_vec(0), Vector3f::UnitZ())
        * AngleAxisf(rot_vec(1), Vector3f::UnitY())
        * AngleAxisf(rot_vec(2), Vector3f::UnitZ());

    temp_vec = sigma_points_propagated[i].segment(0,3);
    sigma_points_propagated[i].segment(0,3) = rot_mat * temp_vec;

    temp_vec = sigma_points_propagated[i].segment(3,3);
    sigma_points_propagated[i].segment(3,3) = rot_mat * temp_vec;

    ++i;
  }
  */
}

/*!
 * \brief meas_function : defines the measurement function which will be used in the UKF
 * \warning Keep the UKF state vector in mind when designing it !
 */
void meas_function(const MatrixXf &vec_in, MatrixXf &vec_measured) {
  vec_measured.resizeLike(vec_in);

  // For now we just keep the points as they are
  for (int i=0; i<vec_in.rows(); ++i) {
    vec_measured(i,0) = vec_in(i,0);
  }
}

/*!
 * \brief meas_q_function : defines the measurement function which will be used in the UKF
 */
void meas_q_function(const Quaternionf &q_in, Quaternionf &q_measured) {
  q_measured = q_in;
}


/*!
 * \brief MotionEstimation::MotionEstimation
 * \param pos
 * \param speed
 */
MotionEstimation::MotionEstimation(const float *variable,
                                   const float ukf_measure_noise = 1.f,
                                   const float ukf_process_noise = 1.f,
                                   const float ukf_kappa = 0.5f) {

  // Constructor for a new pose estimator
  // Contains 2 types of variables :
  // - initial position
  // - initial speed

  _measure.setZero (6,1);
  _measure(0,0) = variable[0];
  _measure(1,0) = variable[1];
  _measure(2,0) = variable[2];
  _measure(3,0) = variable[0];
  _measure(4,0) = variable[1];
  _measure(5,0) = variable[2];


  _initial_cov.setIdentity(6,6);
  _model_noise.setIdentity(6,6);
  _measurement_noise.setIdentity(6,6);

  _initial_cov *= 1.f;
  _model_noise *= ukf_process_noise;
  _measurement_noise *= ukf_measure_noise;

  // Allocate UKF and set propagation function
  filter = new UKF(_measure,
                   _initial_cov,
                   _model_noise,
                   _measurement_noise,
                   ukf_kappa);

  filter->setPropagationFunction (&pointPropagation_speed);
  filter->setMeasurementFunction (&meas_function);

  _measure_latest = _measure;
  _filter_angular_speed = false;
}

/*!
 * \brief MotionEstimation::MotionEstimation
 * \param pos
 * \param speed
 */
MotionEstimation::MotionEstimation(const float *speed,
                                   const float *angular_speed,
                                   const float ukf_measure_noise = 1.f,
                                   const float ukf_measure_q_noise = 1.f,
                                   const float ukf_process_noise = 1.f,
                                   const float ukf_process_q_noise = 1.f,
                                   const float ukf_kappa = 0.f,
                                   const float ukf_kappa_q = 0.f) {

  // Constructor for a new pose estimator
  // Contains 2 types of variables :
  // - initial speed
  // - initial angular speed

  _measure.setZero (6,1);
  _measure(0,0) = speed[0];
  _measure(1,0) = speed[1];
  _measure(2,0) = speed[2];

  _measure(3,0) = angular_speed[0];
  _measure(4,0) = angular_speed[1];
  _measure(5,0) = angular_speed[2];

  // Vector space noise matrices
  _initial_cov.setIdentity(3,3);
  _model_noise.setIdentity(3,3);
  _measurement_noise.setIdentity(3,3);

  _initial_cov.block(0,0,3,3)       *= ukf_measure_noise;
  _model_noise.block(0,0,3,3)       *= ukf_process_noise;
  _measurement_noise.block(0,0,3,3) *= ukf_measure_noise;

  // DEBUG
  // Limit moves on the x/y axis :
  _model_noise.block(0,0,2,2)       /= 6.f;
  // DEBUG

  // Quaternion space noise matrices
  _initial_q_cov.setIdentity(3,3);
  _model_q_noise.setIdentity(3,3);
  _measurement_q_noise.setIdentity(3,3);

  _initial_q_cov *= ukf_measure_q_noise;
  _model_q_noise *= ukf_process_q_noise;
  // DEBUG
  // Limit moves on the x/y axis :
  _model_q_noise.block(0,0,1,1)       /= 3.f;
  _model_q_noise.block(2,2,1,1)       /= 3.f;
  // DEBUG
  _measurement_q_noise *= ukf_measure_q_noise;

  // Allocate UKF and set propagation function
  filter = new UKF(_measure.block(0,0, 3,1),
                   _measure.block(3,0, 3,1),
                   _initial_cov,
                   _initial_q_cov,
                   _model_noise,
                   _model_q_noise,
                   _measurement_noise,
                   _measurement_q_noise,
                   ukf_kappa,
                   ukf_kappa_q);

  filter->setPropagationFunction (&pointPropagation_speed);
  filter->setMeasurementFunction (&meas_function);

  filter->setMeasurementQFunction (&meas_q_function);
  filter->setPropagationQFunction (&pointPropagation_angularSpeed);

  _measure_latest = _measure;
  _filter_angular_speed = true;
}

// TODO: Add a constructor for an estimator taking angles into account !

MotionEstimation::~MotionEstimation () {
  delete filter;

  // Nothing to do for Eigen matrices (?)
}



void MotionEstimation::setMeasurementSettings(const float ukf_measure_noise,
                                              const float ukf_measure_q_noise,
                                              const float ukf_process_noise,
                                              const float ukf_process_q_noise) {

  // Vector space noise matrices
  _model_noise.setIdentity(3,3);
  _measurement_noise.setIdentity(3,3);

  _model_noise.block(0,0,3,3)       *= ukf_process_noise;
  _measurement_noise.block(0,0,3,3) *= ukf_measure_noise;

  filter->setProcessNoise (_model_noise);
  filter->setMeasurementNoise (_measurement_noise);

  // Quaternion space noise matrices
  _model_q_noise.setIdentity(3,3);
  _measurement_q_noise.setIdentity(3,3);

  _model_q_noise *= ukf_process_q_noise;
  _measurement_q_noise *= ukf_measure_q_noise;

  filter->setProcessQNoise (_model_q_noise);
  filter->setMeasurementQNoise (_measurement_q_noise);
}


/*!
 * \brief MotionEstimation::update
 * \param speed
 */
void MotionEstimation::update(const float *variable) {
  _measure_latest(0,0) = variable[0];
  _measure_latest(1,0) = variable[1];
  _measure_latest(2,0) = variable[2];
  _measure_latest(3,0) = variable[3];
  _measure_latest(4,0) = variable[4];
  _measure_latest(5,0) = variable[5];

#ifdef DEBUG
  cout << "\nNew measure\n" << _measure_latest << endl;
#endif

  filter->update(_measure_latest);
}

void MotionEstimation::update(const float *speed,
                              const float *angular_speed) {

  if ((_measure_latest.rows () != 6) ||
      (!_filter_angular_speed)){
    THROW_ERR("Motion filtering : filter initialization went wrong");
  }

  // Get latest updated state :
  MatrixXf lastest_state;
  filter->getStatePost (lastest_state);

  _measure_latest(0,0) = speed[0];
  _measure_latest(1,0) = speed[1];
  _measure_latest(2,0) = speed[2];

  _measure_latest(3,0) = angular_speed[0];
  _measure_latest(4,0) = angular_speed[1];
  _measure_latest(5,0) = angular_speed[2];

  filter->update(_measure_latest.block(0,0,3,1),
                 _measure_latest.block(3,0,3,1));
}



void MotionEstimation::predict() {
  filter->predict();
}

void MotionEstimation::getLatestState(float *state_out) const{

  Eigen::MatrixXf new_state;

  if (!_filter_angular_speed) {
    filter->getStatePost (new_state);

    // TODO: memcpy and not a crappy loop..
    for (unsigned int i=0; i<new_state.rows(); ++i) {
        state_out[i] = new_state(i,0);
      }

  } else {
    new_state.resize (6,1);

    filter->getStatePost (new_state);

    state_out[0] = new_state(0,0);  // Speed
    state_out[1] = new_state(1,0);
    state_out[2] = new_state(2,0);

    state_out[3] = new_state(3,0);  // Angular values
    state_out[4] = new_state(4,0);
    state_out[5] = new_state(5,0);
  }
}

void MotionEstimation::getPropagatedState(float *state_out) const{

  Eigen::MatrixXf new_state;

  if (!_filter_angular_speed) {
    filter->getStatePre(new_state);

    // TODO: memcpy and not a crappy loop..
    for (unsigned int i=0; i<new_state.rows(); ++i) {
        state_out[i] = new_state(i,0);
      }
  } else {
    new_state.resize (6,1);

    filter->getStatePost (new_state);

    state_out[0] = new_state(0,0);  // Speed
    state_out[1] = new_state(1,0);
    state_out[2] = new_state(2,0);

    state_out[3] = new_state(3,0);  // Angular values
    state_out[4] = new_state(4,0);
    state_out[5] = new_state(5,0);
  }
}
