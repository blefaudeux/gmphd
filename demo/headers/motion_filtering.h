#ifndef MOTION_FILTERING_H
#define MOTION_FILTERING_H

#include "unscented_KF.h"

class MotionEstimation
{
  private :
    bool _filter_angular_speed;

    std::vector<float> past_positions;
    std::vector<float[3]> past_speed;

    UKF *filter;

    MatrixXf _measure;
    MatrixXf _initial_cov;
    MatrixXf _model_noise;
    MatrixXf _measurement_noise;
    MatrixXf _measure_latest;

    MatrixXf _initial_q_cov;
    MatrixXf _model_q_noise;
    MatrixXf _measurement_q_noise;

  public:
    float dt;

    /*!
     * \brief MotionEstimation
     * Default constructor : filter motion (vector) only
     * \param speed
     * \param ukf_measure_noise
     * \param ukf_process_noise
     * \param ukf_alpha
     * \param ukf_beta
     */
    MotionEstimation(const float *variable,
                     const float ukf_measure_noise,
                     const float ukf_process_noise,
                     const float ukf_kappa);

    /*!
     * \brief MotionEstimation
     *  Alternate constructor : filter motion taking angles into account
     *
     * \param speed
     * \param angular_speed
     * \param ukf_measure_noise
     * \param ukf_measure_q_noise
     * \param ukf_process_noise
     * \param ukf_process_q_noise
     * \param ukf_alpha     : spread of the vecor sigma points ([0,1])
     * \param ukf_alpha_q   : spread of the quaternion sigma points ([0,1])
     * \param ukf_beta      : weight of the previous mean value in the vector sigma points
     * \param ukf_beta_q    : weight of the previous mean value in the vector sigma points
     */
    MotionEstimation(const float *speed,
                     const float *angular_speed,
                     const float ukf_measure_noise,
                     const float ukf_measure_q_noise,
                     const float ukf_process_noise,
                     const float ukf_process_q_noise,
                     const float ukf_kappa,
                     const float ukf_kappa_q);

    ~MotionEstimation();


    /*!
     * \brief Get the estimated state from the filter
     * \param state_out
     */
    void getLatestState(float *state_out) const;

    /*!
     * \brief Get the estimated state from the filter
     * \param state_out
     */
    void getPropagatedState(float *state_out) const;


    /*!
     * \brief Predict a new state from the previous estimation
     */
    void predict();

    /*!
     * \brief Update the filter settings as regards measurements noise
     * \param ukf_measure_noise
     * \param ukf_measure_q_noise
     * \param ukf_process_noise
     * \param ukf_process_q_noise
     */
    void setMeasurementSettings(const float ukf_measure_noise,
                                const float ukf_measure_q_noise,
                                const float ukf_process_noise,
                                const float ukf_process_q_noise);

    void update(const float *variable);

    void update(const float *speed,
                const float *angular_speed);

};

#endif // POSE_ESTIMATION_H
