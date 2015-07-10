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

    SpawningModel(int dim = 2):
        m_dim(dim)
    {
        m_state = m_dim * 2;
        m_trans = MatrixXf::Ones(m_state, m_state);
        m_cov = MatrixXf::Ones(m_state, m_state);
        m_offset = MatrixXf::Zero(m_state,1);
        m_weight = 0.1f;
    }

    int m_dim;
    int m_state;

    float m_weight;

    MatrixXf m_trans;
    MatrixXf m_cov;
    MatrixXf m_offset;
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

    void  setNewMeasurements(const vector<float> &position,
                             const vector<float> &speed);

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

    void  setBirthModel(vector<GaussianModel> &m_birthModel);

    void  setSpawnModel(vector<SpawningModel> &spawnModels);

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

    MatrixXf  m_tgtDynTrans;
    MatrixXf  m_tgtDynCov;

    MatrixXf  m_obsMat;
    MatrixXf  m_obsMatT;
    MatrixXf  m_obsCov;

    MatrixXf I;

    // Temporary matrices, used for the update process
    vector <MatrixXf, aligned_allocator <MatrixXf> > m_covariance;
    vector <MatrixXf, aligned_allocator <MatrixXf> > m_expMeasure;
    vector <MatrixXf, aligned_allocator <MatrixXf> > m_expDisp;
    vector <MatrixXf, aligned_allocator <MatrixXf> > m_uncertainty;

    GaussianMixture m_birthModel;

    GaussianMixture m_birthTargets;
    GaussianMixture m_currTargets;
    GaussianMixture m_expTargets;
    GaussianMixture m_extractedTargets;
    GaussianMixture m_measTargets;
    GaussianMixture m_spawnTargets;

    /*!
     * \brief The spawning models (how gaussians spawn from existing targets)
     * Example : how airplanes take off from a carrier..
     */
    vector <SpawningModel, aligned_allocator <SpawningModel> > m_spawnModels;

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
