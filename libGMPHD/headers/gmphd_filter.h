#pragma once

// Author : Benjamin Lefaudeux (blefaudeux@github)

#include "gaussian_mixture.h"
#include <iostream>
#include <memory>

using namespace std;
using namespace Eigen;

/*!
 * \brief The spawning_model struct
 */
template <size_t S>
struct SpawningModel
{
  SpawningModel()
  {
    m_trans.setIdentity();
    m_cov.setIdentity();
    m_offset.setZero();
    m_weight = 0.1f;
  }

  float m_weight;

  Matrix<float, S, S> m_trans;
  Matrix<float, S, S> m_cov;
  Matrix<float, S, 1> m_offset;
};

/*!
 * \brief The gmphd_filter class
 */
template <size_t D>
class GMPHD
{
  static const size_t S = D * 2;

public:
  struct Target
  {
    Matrix<float, D, 1> position;
    Matrix<float, D, 1> speed;
    float weight;
  };

  GMPHD(int max_gaussians, bool verbose = false) : m_maxGaussians(max_gaussians), m_bVerbose(verbose)

  {
    m_dimMeasures = D;
    m_dimState = 2 * m_dimMeasures;
    m_pruneTruncThld = 0.f;
    m_pDetection = 0.f;
    m_pSurvival = 0.f;

    // Initialize all gaussian mixtures, we know the dimension now
    m_measTargets.reset(new GaussianMixture<S>());
    m_birthTargets.reset(new GaussianMixture<S>());
    m_currTargets.reset(new GaussianMixture<S>());
    m_expTargets.reset(new GaussianMixture<S>());
    m_extractedTargets.reset(new GaussianMixture<S>());
    m_spawnTargets.reset(new GaussianMixture<S>());
  }

  bool isInitialized()
  {
    if (m_tgtDynTrans.cols() != m_dimState)
    {
      printf("[GMPHD] - Motion model not set\n");
      return false;
    }

    if (m_pruneTruncThld <= 0.f)
    {
      printf("[GMPHD] - Pruning parameters not set\n");
      return false;
    }

    if (m_pDetection <= 0.f || m_pSurvival <= 0.f)
    {
      printf("[GMPHD] - Observation model not set\n");
      return false;
    }

    return true;
  }

  // Input: raw measurements and possible ref change
  void setNewReferential(Matrix4f const &transform)
  {
    // Change referential for every gaussian in the gaussian mixture
    m_currTargets->changeReferential(transform);
  }

  void setNewMeasurements(vector<Target> const &measurements)
  {
    // Clear the gaussian mixture
    m_measTargets->m_gaussians.clear();

    for (const auto &meas : measurements)
    {
      // Create new gaussian model according to measurement
      GaussianModel<S> new_obs;
      new_obs.m_mean.head(m_dimMeasures) = meas.position;
      new_obs.m_mean.tail(m_dimMeasures) = meas.speed;
      new_obs.m_cov = m_obsCov;
      new_obs.m_weight = meas.weight;

      m_measTargets->m_gaussians.push_back(std::move(new_obs));
    }
  }

  // Output
  vector<Target> getTrackedTargets(float const &extract_thld)
  {
    // TODO: Rewrite in modern C++

    // Fill in "extracted_targets" from the "current_targets"
    extractTargets(extract_thld);

    vector<Target> targets;

    for (auto const &gaussian : m_extractedTargets->m_gaussians)
    {
      targets.push_back({.position = gaussian.m_mean.head(m_dimMeasures), .speed = gaussian.m_mean.tail(m_dimMeasures), .weight = gaussian.m_weight});
    }
    return targets;
  }

  // Parameters to set before use
  void setDynamicsModel(float sampling, float processNoise)
  {
    m_samplingPeriod = sampling;
    m_processNoise = processNoise;

    // Fill in propagation matrix :
    m_tgtDynTrans.setIdentity();

    for (unsigned int i = 0; i < m_dimMeasures; ++i)
    {
      m_tgtDynTrans(i, m_dimMeasures + i) = m_samplingPeriod;
    }

    // Fill in covariance matrix
    // Extra covariance added by the dynamics. Could be 0.
    m_tgtDynCov = processNoise * processNoise * Matrix<float, S, S>::Identity();
  }

  void setDynamicsModel(MatrixXf const &tgt_dyn_transitions, MatrixXf const &tgt_dyn_covariance)
  {
    m_tgtDynTrans = tgt_dyn_transitions;
    m_tgtDynCov = tgt_dyn_covariance;
  }

  void setSurvivalProbability(float _prob_survival)
  {
    m_pSurvival = _prob_survival;
  }

  void setObservationModel(float probDetectionOverall, float measNoisePose,
                           float measurement_noise_speed, float measNoiseBackground)
  {
    m_pDetection = probDetectionOverall;
    m_measNoisePose = measNoisePose;
    m_measNoiseSpeed = measurement_noise_speed;
    m_measNoiseBackground = measNoiseBackground; // False detection probability

    // Set model matrices
    m_obsMat.setIdentity();
    m_obsMatT = m_obsMat.transpose();
    m_obsCov.setIdentity();

    // FIXME: deal with the _motion_model parameter !
    m_obsCov.block(0, 0, m_dimMeasures, m_dimMeasures) *= m_measNoisePose * m_measNoisePose;
    m_obsCov.block(m_dimMeasures, m_dimMeasures, m_dimMeasures, m_dimMeasures) *= m_measNoiseSpeed * m_measNoiseSpeed;
  }

  void setPruningParameters(float prune_trunc_thld, float prune_merge_thld,
                            int prune_max_nb)
  {
    m_pruneTruncThld = prune_trunc_thld;
    m_pruneMergeThld = prune_merge_thld;
    m_nMaxPrune = prune_max_nb;
  }

  void setBirthModel(const GaussianMixture<S> &birthModel)
  {
    m_birthModel.reset(new GaussianMixture<S>(birthModel));
  }

  void setSpawnModel(vector<SpawningModel<S>> &spawnModels)
  {
    m_spawnModels = spawnModels;
  }

  void propagate()
  {
    m_nPredTargets = 0;

    // Predict new targets (spawns):
    predictBirth();

    // Predict propagation of expected targets :
    predictTargets();

    // Build the update components
    buildUpdate();

    // Update GMPHD
    update();

    // Prune gaussians (remove weakest, merge close enough gaussians)
    pruneGaussians();

    // Clean vectors :
    m_expMeasure.clear();
    m_expDisp.clear();
    m_uncertainty.clear();
    m_covariance.clear();
  }

  void reset()
  {
    m_currTargets->m_gaussians.clear();
    m_extractedTargets->m_gaussians.clear();
  }

private:
  /*!
     * \brief The spawning models (how gaussians spawn from existing targets)
     * Example : how airplanes take off from a carrier..
     */
  vector<SpawningModel<S>> m_spawnModels;

  void buildUpdate()
  {

    // Concatenate all the wannabe targets :
    // - birth targets
    m_iBirthTargets.clear();

    if (m_birthTargets->m_gaussians.size() > 0)
    {
      for (unsigned int i = 0; i < m_birthTargets->m_gaussians.size(); ++i)
      {
        m_iBirthTargets.push_back(m_expTargets->m_gaussians.size() + i);
      }

      m_expTargets->m_gaussians.insert(m_expTargets->m_gaussians.end(), m_birthTargets->m_gaussians.begin(),
                                       m_birthTargets->m_gaussians.begin() + m_birthTargets->m_gaussians.size());
    }

    // - spawned targets
    if (m_spawnTargets->m_gaussians.size() > 0)
    {
      m_expTargets->m_gaussians.insert(m_expTargets->m_gaussians.end(), m_spawnTargets->m_gaussians.begin(),
                                       m_spawnTargets->m_gaussians.begin() + m_spawnTargets->m_gaussians.size());
    }

    // Compute PHD update components (for every expected target)
    m_nPredTargets = m_expTargets->m_gaussians.size();

    m_expMeasure.clear();
    m_expMeasure.reserve(m_nPredTargets);

    m_expDisp.clear();
    m_expDisp.reserve(m_nPredTargets);

    m_uncertainty.clear();
    m_uncertainty.reserve(m_nPredTargets);

    m_covariance.clear();
    m_covariance.reserve(m_nPredTargets);

    for (auto const &tgt : m_expTargets->m_gaussians)
    {
      // Compute the expected measurement
      m_expMeasure.push_back(m_obsMat * tgt.m_mean);
      m_expDisp.push_back(m_obsCov + m_obsMat * tgt.m_cov * m_obsMatT);

      m_uncertainty.push_back(tgt.m_cov * m_obsMatT * m_expDisp.back().inverse());
      m_covariance.push_back((Matrix<float, S, S>::Identity() - m_uncertainty.back() * m_obsMat) * tgt.m_cov);
    }
  }

  void extractTargets(float threshold)
  {
    // TODO: rewrite in modern C++

    float const thld = std::max(threshold, 0.f);

    // Get trough every target, keep the ones whose weight is above threshold
    m_extractedTargets->m_gaussians.clear();

    for (auto const &current_target : m_currTargets->m_gaussians)
    {
      if (current_target.m_weight >= thld)
      {
        m_extractedTargets->m_gaussians.push_back(current_target);
      }
    }
  }

  void predictBirth()
  {
    m_spawnTargets->m_gaussians.clear();
    m_birthTargets->m_gaussians.clear();

    // -----------------------------------------
    // Compute spontaneous births
    m_birthTargets->m_gaussians = m_birthModel->m_gaussians;
    m_nPredTargets += m_birthTargets->m_gaussians.size();

    // -----------------------------------------
    // Compute spawned targets
    for (auto const &curr : m_currTargets->m_gaussians)
    {
      for (auto const &spawn : m_spawnModels)
      {
        GaussianModel<S> new_spawn;

        // Define a gaussian model from the existing target
        // and spawning properties
        new_spawn.m_weight = curr.m_weight * spawn.m_weight;

        new_spawn.m_mean = spawn.m_offset + spawn.m_trans * curr.m_mean;

        new_spawn.m_cov = spawn.m_cov + spawn.m_trans * curr.m_cov * spawn.m_trans.transpose();

        // Add this new gaussian to the list of expected targets
        m_spawnTargets->m_gaussians.push_back(std::move(new_spawn));

        // Update the number of expected targets
        ++m_nPredTargets;
      }
    }
  }

  void predictTargets()
  {
    m_expTargets->m_gaussians.clear();
    m_expTargets->m_gaussians.reserve(m_currTargets->m_gaussians.size());

    for (auto const &curr : m_currTargets->m_gaussians)
    {
      // Compute the new shape of the target
      GaussianModel<S> new_target;
      new_target.m_weight = m_pSurvival * curr.m_weight;
      new_target.m_mean = m_tgtDynTrans * curr.m_mean;
      new_target.m_cov = m_tgtDynCov + m_tgtDynTrans * curr.m_cov * m_tgtDynTrans.transpose();

      // Push back to the expected targets
      m_expTargets->m_gaussians.push_back(new_target);
      ++m_nPredTargets;
    }
  }

  void pruneGaussians()
  {
    m_currTargets->prune(m_pruneTruncThld, m_pruneMergeThld, m_nMaxPrune);
  }

  void update()
  {
    unsigned int n_meas, n_targt, index;
    m_currTargets->m_gaussians.clear();

    // We'll consider every possible association : vector size is (expected targets)*(measured targets)
    m_currTargets->m_gaussians.resize((m_measTargets->m_gaussians.size() + 1) *
                                      m_expTargets->m_gaussians.size());

    // First set of gaussians : mere propagation of existing ones
    // \warning : don't propagate the "birth" targets...
    // we set their weight to 0

    m_nPredTargets = m_expTargets->m_gaussians.size();
    int i_birth_current = 0;

    for (unsigned int i = 0; i < m_nPredTargets; ++i)
    {
      if (i != m_iBirthTargets[i_birth_current])
      {
        m_currTargets->m_gaussians[i].m_weight = (1.f - m_pDetection) *
                                                 m_expTargets->m_gaussians[i].m_weight;
      }
      else
      {
        i_birth_current = std::min(i_birth_current + 1, (int)m_iBirthTargets.size());
        m_currTargets->m_gaussians[i].m_weight = 0.f;
      }

      m_currTargets->m_gaussians[i].m_mean = m_expTargets->m_gaussians[i].m_mean;
      m_currTargets->m_gaussians[i].m_cov = m_expTargets->m_gaussians[i].m_cov;
    }

    // Second set of gaussians : match observations and previsions
    if (m_measTargets->m_gaussians.size() == 0)
    {
      return;
    }

    // FIXME: Rewrite in modern C++..

    for (n_meas = 1; n_meas <= m_measTargets->m_gaussians.size(); ++n_meas)
    {
      for (n_targt = 0; n_targt < m_nPredTargets; ++n_targt)
      {
        index = n_meas * m_nPredTargets + n_targt;

        // Compute matching factor between predictions and measures.
        const auto distance = mahalanobis<2>(m_measTargets->m_gaussians[n_meas - 1].m_mean.block(0, 0, m_dimMeasures, 1),
                                             m_expMeasure[n_targt].block(0, 0, m_dimMeasures, 1),
                                             m_expDisp[n_targt].block(0, 0, m_dimMeasures, m_dimMeasures));

        m_currTargets->m_gaussians[index].m_weight = m_pDetection * m_expTargets->m_gaussians[n_targt].m_weight / distance;

        m_currTargets->m_gaussians[index].m_mean = m_expTargets->m_gaussians[n_targt].m_mean +
                                                   m_uncertainty[n_targt] * (m_measTargets->m_gaussians[n_meas - 1].m_mean - m_expMeasure[n_targt]);

        m_currTargets->m_gaussians[index].m_cov = m_covariance[n_targt];
      }

      // Normalize weights in the same predicted set,
      // taking clutter into account
      m_currTargets->normalize(m_measNoiseBackground, n_meas * m_nPredTargets,
                               (n_meas + 1) * m_nPredTargets, 1);
    }
  }

private:
  bool m_motionModel;
  bool m_bVerbose;

  uint m_maxGaussians;
  uint m_dimMeasures;
  uint m_dimState;
  uint m_nPredTargets;
  uint m_nCurrentTargets;
  uint m_nMaxPrune;

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

  Matrix<float, S, S> m_tgtDynTrans;
  Matrix<float, S, S> m_tgtDynCov;

  Matrix<float, S, S> m_obsMat;
  Matrix<float, S, S> m_obsMatT;
  Matrix<float, S, S> m_obsCov;

  // Temporary matrices, used for the update process
  vector<Matrix<float, S, S>> m_covariance;
  vector<Matrix<float, S, 1>> m_expMeasure;
  vector<Matrix<float, S, S>> m_expDisp;
  vector<Matrix<float, S, S>> m_uncertainty;

  std::unique_ptr<GaussianMixture<S>> m_birthModel;

  std::unique_ptr<GaussianMixture<S>> m_birthTargets;
  std::unique_ptr<GaussianMixture<S>> m_currTargets;
  std::unique_ptr<GaussianMixture<S>> m_expTargets;
  std::unique_ptr<GaussianMixture<S>> m_extractedTargets;
  std::unique_ptr<GaussianMixture<S>> m_measTargets;
  std::unique_ptr<GaussianMixture<S>> m_spawnTargets;
};