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

    Matrix<float, 6,6> m_trans; // Transition matrix
    Matrix<float, 6,6> m_cov;
    Matrix<float, 6,1> m_offset;

    float m_weight;
};


/*!
 * \brief The gmphd_filter class
 */
class GMPHD
{
public:
    /*!
     * \brief GmphdFilter
     * \param max_gaussians : max number of evaluated targets
     * \param dimension     : dimension of the measurement vector
     * \param motion_model  : intrinsic (constant speed) motion model
     * \param verbose       : print debug
     */
    GMPHD(int max_gaussians,
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

    void  print();

    void  propagate();

    void  reset();

    void  setNewReferential(const Matrix4f *transform);

    void  setNewMeasurements(vector<float> &position,
                             vector<float> &speed);

    void  setDynamicsModel(float _sampling,
                           float m_processNoise);

    void setDynamicsModel(MatrixXf &tgt_dyn_transitions,
                          MatrixXf &tgt_dyn_covariance);

    void  setSurvivalProbability(float _prob_survival);

    void  setObservationModel(float _prob_detection_overall,
                              float m_measNoisePose,
                              float m_measNoiseSpeed,
                              float m_measNoiseBackground);

    void  setPruningParameters(float  prune_trunc_thld,
                               float  prune_merge_thld,
                               int    prune_max_nb);

    void  setBirthModel(vector<GaussianModel> &_birth_model);

    void  setSpawnModel(vector <SpawningModel, aligned_allocator<SpawningModel> > &_spawn_models);

private:
    int   m_maxGaussians;
    int   m_dimMeasures;
    bool  m_motionModel;
    bool  m_bVerbose;
    int   m_dimState;
    int   m_nPredictedTargets;
    int   m_nCurrentTargets;
    int   m_nMaxPrune;

    float m_pSurvival;
    float m_pDetection;

    float m_samplingPeriod;
    float m_processNoise;

    float m_pruneMergeThld;
    float m_pruneTruncThld;

    float m_measNoisePose;
    float m_measNoiseSpeed;
    float m_measNoiseBackground; // Background detection "noise", other models are possible..

    vector<int> m_iBirthTargets;

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
