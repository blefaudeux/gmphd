#ifndef GMPHD_FILTER_H
#define GMPHD_FILTER_H

// Author : Benjamin Lefaudeux (blefaudeux@github)


#include "gaussian_mixture.h"
#include <iostream>
#include <memory>

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

typedef uint uint;

/*!
 * \brief The gmphd_filter class
 */
class GMPHD
{
public:
  GMPHD(int max_gaussians, int dimension,
        bool motion_model = false, bool verbose = false);

  bool isInitialized();

  // Input: raw measurements and possible ref change
  void  setNewReferential( Matrix4f const & transform);

  void  setNewMeasurements( vector<float> const & position, vector<float> const & speed);

  // Output
  void  getTrackedTargets( vector<float> & position, vector<float> & speed, vector<float> & weight,
                           float const & extract_thld );

  // Parameters to set before use
  void  setDynamicsModel( float sampling, float processNoise );

  void  setDynamicsModel( MatrixXf const & tgt_dyn_transitions, MatrixXf const & tgt_dyn_covariance);

  void  setSurvivalProbability(float _prob_survival);

  void  setObservationModel(float probDetectionOverall, float m_measNoisePose,
                            float m_measNoiseSpeed, float m_measNoiseBackground );

  void  setPruningParameters(float  prune_trunc_thld, float  prune_merge_thld,
                             int    prune_max_nb);

  void  setBirthModel(vector<GaussianModel> & m_birthModel);

  void  setSpawnModel(vector<SpawningModel> & spawnModels);

  // Auxiliary functions
  void  print() const;

  void  propagate();

  void  reset();

private:
  /*!
     * \brief The spawning models (how gaussians spawn from existing targets)
     * Example : how airplanes take off from a carrier..
     */
  vector <SpawningModel, aligned_allocator <SpawningModel> > m_spawnModels;

  void  buildUpdate();

  void  extractTargets(float threshold);

  void  predictBirth();

  void  predictTargets();

  void  pruneGaussians();

  void  update();


private:
  bool  m_motionModel;
  bool  m_bVerbose;

  uint   m_maxGaussians;
  uint   m_dimMeasures;
  uint   m_dimState;
  uint   m_nPredTargets;
  uint   m_nCurrentTargets;
  uint   m_nMaxPrune;

  float m_pSurvival;
  float m_pDetection;

  float m_samplingPeriod;
  float m_processNoise;

  float m_pruneMergeThld;
  float m_pruneTruncThld;

  float m_measNoisePose;
  float m_measNoiseSpeed;
  float m_measNoiseBackground; // Background detection "noise", other models are possible..

  vector<uint> m_iBirthTargets;

  MatrixXf  m_tgtDynTrans;
  MatrixXf  m_tgtDynCov;

  MatrixXf  m_obsMat;
  MatrixXf  m_obsMatT;
  MatrixXf  m_obsCov;

  // Temporary matrices, used for the update process
  vector <MatrixXf, aligned_allocator <MatrixXf> > m_covariance;
  vector <MatrixXf, aligned_allocator <MatrixXf> > m_expMeasure;
  vector <MatrixXf, aligned_allocator <MatrixXf> > m_expDisp;
  vector <MatrixXf, aligned_allocator <MatrixXf> > m_uncertainty;

  std::unique_ptr<GaussianMixture> m_birthModel;

  std::unique_ptr<GaussianMixture> m_birthTargets;
  std::unique_ptr<GaussianMixture> m_currTargets;
  std::unique_ptr<GaussianMixture> m_expTargets;
  std::unique_ptr<GaussianMixture> m_extractedTargets;
  std::unique_ptr<GaussianMixture> m_measTargets;
  std::unique_ptr<GaussianMixture> m_spawnTargets;

private:

  template <size_t D>
  float mahalanobis(const Matrix <float, D,1> &point,
                    const Matrix <float, D,1> &mean,
                    const Matrix <float, D,D> &cov)
  {
      int ps = point.rows();
      MatrixXf x_cen = point-mean;
      MatrixXf b = MatrixXf::Identity(ps,ps);

      // TODO: Ben - cov needs to be normalized !
      cov.ldlt().solveInPlace(b);
      x_cen = b*x_cen;
      MatrixXf res = x_cen.transpose() * x_cen;
      return res.sum();
  }

  template <size_t D>
  float   gaussDensity(const Matrix <float, D,1> &point,
                       const Matrix <float, D,1> &mean,
                       const Matrix <float, D,D> &cov) const
  {
    float det, res;

    Matrix <float, D, D> cov_inverse;
    Matrix <float, D, 1> mismatch;

    det = cov.determinant();
    cov_inverse = cov.inverse();

    mismatch = point - mean;

    Matrix <float, 1, 1> distance = mismatch.transpose() * cov_inverse * mismatch;

    distance /= -2.f;

    // Deal with faulty determinant case
    if (det == 0.f)
    {
      return 0.f;
    }

    res = 1.f/sqrt(pow(2*M_PI, D) * fabs(det)) * exp(distance.coeff (0,0));

    if (isinf(det))
    {
      printf("Problem in multivariate gaussian\n distance : %f - det %f\n", distance.coeff (0,0), det);
      cout << "Cov \n" << cov << endl << "Cov inverse \n" << cov_inverse << endl;
      return 0.f;
    }

    return res;
  }
};

#endif // GMPHD_FILTER_H
