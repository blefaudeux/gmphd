#ifndef GMPHD_FILTER_H
#define GMPHD_FILTER_H

// Author : Benjamin Lefaudeux (blefaudeux@github)


#include "gaussian_mixture.h"
#include  <iostream>

using namespace std;
using namespace Eigen;

/*!
 * \brief The spawning_model struct
 */
struct SpawningModel {
  /*!
     * \brief transition matrix for the linear modeling
     * of gaussian model motion
     * (one for every spawning model !)
     */
    Matrix<float, 6,6> trans;


    /*!
     * \brief covariance matrix for the spawned gaussians
     * from the initial gaussian model
     */
    Matrix<float, 6,6> cov;

    /*!
     * \brief offset // TODO : define properly...
     */
    Matrix<float, 6,1> offset;

    /*!
     * \brief the weight of the spawned gaussian
     */
    float weight;
};


/*!
 * \brief The gmphd_filter class
 */
class GmphdFilter
{
public:
    /*!
     * \brief GmphdFilter
     * \param max_gaussians : max number of evaluated targets
     * \param dimension     : dimension of the measurement vector
     * \param motion_model  : intrinsic (constant speed) motion model
     * \param verbose       : print debug
     */
    GmphdFilter(int max_gaussians,
                int dimension,
                bool motion_model = false,
                bool verbose = false);

    /*!
     * \brief getTrackedTargets
     * \param positions
     * \param speed
     */
    void  getTrackedTargets(const float extract_thld,
                            vector<float> &position,
                            vector<float> &speed,
                            vector<float> &weight);

    /*!
     * \brief print all the Gaussians on stdout
     */
    void  print();

    /*!
     * \brief The whole propagation step
     * \warning the new measurements must be defined beforehand
     */
    void  propagate();

    /*!
     * \brief reset : remove every item in the filter
     */
    void  reset();

    /*!
     * \brief setNewReferential : set new referential
     * \param transform : transformation (Rotation + Translation)
     * in a classic 4x4 matrix to get from the previous referential
     * to the new one
     */
    void  setNewReferential(const Matrix4f *transform);

    /*!
     * \brief setNewMeasurements
     * \param positions
     * \param speed
     */
    void  setNewMeasurements(vector<float> &position,
                             vector<float> &speed);

     /*!
     * \brief setDynamicsModel
     * \param sampling : time step of the sampling rate
     * \param process_noise
     */
    void  setDynamicsModel(float _sampling,
                           float _process_noise);


    /*!
     * \brief setDynamicsModel
     * \param tgt_dyn_transitions
     * \param tgt_dyn_covariance
     */
    void setDynamicsModel(MatrixXf &tgt_dyn_transitions,
                          MatrixXf &tgt_dyn_covariance);

    /*!
     * \brief setSurvivalProbability
     * \param _prob_survival
     */
    void  setSurvivalProbability(float _prob_survival);

    /*!
     * \brief setObservationModel
     * \param _prob_detection_overall
     * \param _measurement_noise_pose : noise over the position observations
     * \param _measurement_noise_speed : noise over the speed observations
     * \param _measurement_background : false detection probability
     */
    void  setObservationModel(float _prob_detection_overall,
                              float _measurement_noise_pose,
                              float _measurement_noise_speed,
                              float _measurement_background);

    /*!
     * \brief setPruningParameters : parameters which decide for the merge of close gaussians
     * \param _prune_trunc_thld
     * \param _prune_merge_thld
     * \param _prune_max_nb
     */
    void  setPruningParameters(float  prune_trunc_thld,
                               float  prune_merge_thld,
                               int    prune_max_nb);

    /*!
     * \brief setBirthModel : set the model for spontaneous births
     * \param pose
     * \param std
     */
    void  setBirthModel(vector<GaussianModel> &_birth_model);

    /*!
     * \brief setSpawnModel  : set spawning parameters from existing targets,
     * that is : how can sub-targets appear from existing ones (pedestrian
     * spinning off from a group, ...)
     */
    void  setSpawnModel(vector <SpawningModel, aligned_allocator<SpawningModel> > &_spawn_models);

private:
    bool  _verbose;
    bool  _motion_model;
    int   _dim_measures;
    int   _dim_state;
    int   _n_predicted_targets;
    int   _n_current_targets;
    int   _n_max_gaussians;
    int   _prune_max_nb;

    float _p_survival_overall;
    float _p_detection_overall;

    float _sampling_period;
    float _process_noise;

    float _prune_merge_thld;
    float _prune_trunc_thld;

    float _measurement_noise_pose;
    float _measurement_noise_speed;
    float _measurement_background; // Background detection "noise", other models are possible..

    vector<int> _i_birth_targets;

    MatrixXf  _tgt_dyn_transitions;
    MatrixXf  _tgt_dyn_covariance;

    MatrixXf  _obs_matrix;
    MatrixXf  _obs_matrix_T;
    MatrixXf  _obs_covariance;

    MatrixXf I;

    // Temporary matrices, used for the update process
    vector <MatrixXf, aligned_allocator <MatrixXf> > _covariance;
    vector <MatrixXf, aligned_allocator <MatrixXf> > _expected_measure;
    vector <MatrixXf, aligned_allocator <MatrixXf> > _expected_dispersion;
    vector <MatrixXf, aligned_allocator <MatrixXf> > _uncertainty;

    GaussianMixture _birth_model;
    GaussianMixture _birth_targets;
    GaussianMixture _current_targets;
    GaussianMixture _expected_targets;
    GaussianMixture _extracted_targets;
    GaussianMixture _meas_targets;
    GaussianMixture _spawned_targets;

    /*!
     * \brief The spawning models (how gaussians spawn from existing targets)
     * Example : how airplanes take off from a carrier..
     */
    vector <SpawningModel, aligned_allocator <SpawningModel> > _spawn_model;

    /*!
     * \brief Build the update components
     */
    void  buildUpdate();


    /*!
     * \brief gauss_density : multivariate normal distribution. We suppose cov matrix
     * is definite positive, degenerate case is not dealt with.
     * \param point
     * \param mean
     * \param cov
     * \param res
     */
    float gaussDensity(const MatrixXf &point,
                       const MatrixXf &mean,
                       const MatrixXf &cov);

    /*!
     * \brief gauss_density : multivariate normal distribution. We suppose cov matrix
     * is definite positive, degenerate case is not dealt with.
     * \param point
     * \param mean
     * \param cov
     * \param res
     */
    float   gaussDensity_3D(const Matrix <float, 3,1> &point,
                            const Matrix <float, 3,1> &mean,
                            const Matrix <float, 3,3> &cov);

    /*!
     * \brief extractTargets :
     */
    void extractTargets(float threshold);

    /*!
     * \brief Predict birth targets, meaning spontaneous births
     * and spawned targets. Fills in
     * - spawned_targets
     * - birth-targets
     */
    void  predictBirth();

    /*!
     * \brief Predict existing targets propagation.
     * Fills in :
     * - expected_targets
     */
    void  predictTargets();

    /*!
     * \brief Remove and merge gaussians, prune gaussian mixtures
     */
    void  pruneGaussians();

    /*!
     * \brief Update the filter
     */
    void  update();

};

#endif // GMPHD_FILTER_H
