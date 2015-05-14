#include "gmphd_filter.h"


// Author : Benjamin Lefaudeux (blefaudeux@github)


GmphdFilter::GmphdFilter(int max_gaussians,
                         int dimension,
                         bool motion_model,
                         bool verbose):_n_max_gaussians(max_gaussians),
                                      _dim_measures(dimension),
                                      _motion_model(motion_model),
                                      _verbose(verbose){

  if (!motion_model)
    _dim_state = _dim_measures;
  else
    _dim_state = 2 * _dim_measures;

  // Initiate matrices :
  I = MatrixXf::Identity(_dim_state, _dim_state);
}

void  GmphdFilter::buildUpdate () {

  MatrixXf temp_matrix(_dim_state, _dim_state);

  // Concatenate all the wannabe targets :
  // - birth targets
  _i_birth_targets.clear();
  if(_birth_targets.m_gaussians.size () > 0) {
    for (int i=0; i<_birth_targets.m_gaussians.size (); ++i) {
      _i_birth_targets.push_back (_expected_targets.m_gaussians.size () + i);
    }

    _expected_targets.m_gaussians.insert (_expected_targets.m_gaussians.end (),
                               _birth_targets.m_gaussians.begin (),
                               _birth_targets.m_gaussians.begin () + _birth_targets.m_gaussians.size ());
  }

  // - spawned targets
  if (_spawned_targets.m_gaussians.size () > 0) {
    _expected_targets.m_gaussians.insert (_expected_targets.m_gaussians.end (),
                               _spawned_targets.m_gaussians.begin (),
                               _spawned_targets.m_gaussians.begin () + _spawned_targets.m_gaussians.size ());
  }

  if (_verbose) {
    printf("GMPHD : inserted %d birth targets, now %d expected\n",
           _birth_targets.m_gaussians.size (),
           _expected_targets.m_gaussians.size());
           _birth_targets.print ();

    printf("GMPHD : inserted %d spawned targets, now %d expected\n",
           _spawned_targets.m_gaussians.size (),
           _expected_targets.m_gaussians.size());
           _spawned_targets.print ();
    }

  // Compute PHD update components (for every expected target)
  _n_predicted_targets = _expected_targets.m_gaussians.size ();

  _expected_measure.resize (_n_predicted_targets);
  _expected_dispersion.resize (_n_predicted_targets);
  _uncertainty.resize (_n_predicted_targets);
  _covariance.resize (_n_predicted_targets);

  for (int i=0; i< _n_predicted_targets; ++i) {
    // Compute the expected measurement
    _expected_measure[i] = _obs_matrix *
                           _expected_targets.m_gaussians[i].mean;

    _expected_dispersion[i] = _obs_covariance +
                              _obs_matrix * _expected_targets.m_gaussians[i].cov *
                              _obs_matrix_T;

    if (isnan(_expected_dispersion[i](0,0))) {
      printf("NaN value in dispersion\n");
      cout << "Expected cov \n" << _expected_targets.m_gaussians[i].cov << endl << endl;
      THROW_ERR("NaN in GMPHD Update process");
    }

    temp_matrix = _expected_dispersion[i].inverse();

    _uncertainty[i] =  _expected_targets.m_gaussians[i].cov *
                       _obs_matrix_T *
                       temp_matrix;

    _covariance[i] = (I - _uncertainty[i]*_obs_matrix)
                     * _expected_targets.m_gaussians[i].cov;

  }
}

void    GmphdFilter::extractTargets(float threshold) {
  // Deal with crappy settings
  float thld = max(threshold, 0.f);

  // Get trough every target, keep the ones whose weight is above threshold
  _extracted_targets.m_gaussians.clear();

  for (int i=0; i<_current_targets.m_gaussians.size(); ++i)  {
    if (_current_targets.m_gaussians[i].weight >= thld) {
      _extracted_targets.m_gaussians.push_back(_current_targets.m_gaussians[i]);
    }
  }

  printf("GMPHD_extract : %d targets\n", _extracted_targets.m_gaussians.size ());
}

void    GmphdFilter::getTrackedTargets(const float extract_thld,
                                       vector<float> &position,
                                       vector<float> &speed,
                                       vector<float> &weight) {

  // Fill in "extracted_targets" from the "current_targets"
  this->extractTargets(extract_thld);

  position.clear();
  speed.clear();
  weight.clear();

  for (int i=0; i< this->_extracted_targets.m_gaussians.size(); ++i) {
    position.push_back(this->_extracted_targets.m_gaussians[i].mean(0,0));
    position.push_back(this->_extracted_targets.m_gaussians[i].mean(1,0));
    position.push_back(this->_extracted_targets.m_gaussians[i].mean(2,0));

    speed.push_back(this->_extracted_targets.m_gaussians[i].mean(3,0));
    speed.push_back(this->_extracted_targets.m_gaussians[i].mean(4,0));
    speed.push_back(this->_extracted_targets.m_gaussians[i].mean(5,0));

    weight.push_back(this->_extracted_targets.m_gaussians[i].weight);
  }
}

/*!
 * \brief GmphdFilter::gaussDensity
 * \param point
 * \param mean
 * \param cov
 * \return
 *
 *  \warning : we only take the measure space into account for proximity measures !
 *
 */
float   GmphdFilter::gaussDensity(const MatrixXf &point,
                                  const MatrixXf &mean,
                                  const MatrixXf &cov) {

  MatrixXf cov_inverse;
  MatrixXf mismatch;

  float det, res, dist;

  det         = cov.block(0, 0, _dim_measures, _dim_measures).determinant();
  cov_inverse = cov.block(0, 0, _dim_measures, _dim_measures).inverse();
  mismatch    = point.block(0,0,_dim_measures, 1) - mean.block(0,0,_dim_measures,1);

  Matrix <float,1,1> distance = mismatch.transpose() * cov_inverse * mismatch;

  distance /= -2.f;

  dist =  (float) distance(0,0);
  res = 1.f/(pow(2*M_PI, _dim_measures) * sqrt(fabs(det)) * exp(dist));

  return res;
}

float   GmphdFilter::gaussDensity_3D(const Matrix <float, 3,1> &point,
                                     const Matrix <float, 3,1> &mean,
                                     const Matrix <float, 3,3> &cov) {


  float det, res;

  Matrix <float, 3, 3> cov_inverse;
  Matrix <float, 3, 1> mismatch;

  det = cov.determinant();
  cov_inverse = cov.inverse();

  mismatch = point - mean;

  Matrix <float, 1, 1> distance = mismatch.transpose() * cov_inverse * mismatch;

  distance /= -2.f;

  // Deal with faulty determinant case
  if (det == 0.f)
    return 0.f;

  res = 1.f/sqrt(pow(2*M_PI, 3) * fabs(det)) * exp(distance.coeff (0,0));

  if (isinf(det)) {
    printf("Problem in multivariate gaussian\n distance : %f - det %f\n", distance.coeff (0,0), det);
    cout << "Cov \n" << cov << endl << "Cov inverse \n" << cov_inverse << endl;
    return 0.f;
  }

  return res;
}


void  GmphdFilter::predictBirth() {

  _spawned_targets.m_gaussians.clear ();
  _birth_targets.m_gaussians.clear ();

  // -----------------------------------------
  // Compute spontaneous births
  _birth_targets.m_gaussians = _birth_model.m_gaussians;
  _n_predicted_targets += _birth_targets.m_gaussians.size ();

  // -----------------------------------------
  // Compute spawned targets
  GaussianModel new_spawn;

  for (int k=0; k< _current_targets.m_gaussians.size (); ++k) {
    for (int i=0; i< _spawn_model.size (); ++i) {
      // Define a gaussian model from the existing target
      // and spawning properties
      new_spawn.weight = _current_targets.m_gaussians[k].weight
                         * _spawn_model[i].weight;

      new_spawn.mean = _spawn_model[i].offset
                       + _spawn_model[i].trans * _current_targets.m_gaussians[k].mean;

      new_spawn.cov = _spawn_model[i].cov
                      + _spawn_model[i].trans *
                      _current_targets.m_gaussians[k].cov *
                      _spawn_model[i].trans.transpose();

      // Add this new gaussian to the list of expected targets
      _spawned_targets.m_gaussians.push_back (new_spawn);

      // Update the number of expected targets
      _n_predicted_targets ++;
    }
  }
}

void  GmphdFilter::predictTargets () {
  GaussianModel new_target;

  _expected_targets.m_gaussians.resize(_current_targets.m_gaussians.size ());

  for (int i=0; i<_current_targets.m_gaussians.size (); ++i) {
    // Compute the new shape of the target
    new_target.weight = _p_survival_overall *
                        _current_targets.m_gaussians[i].weight;

    new_target.mean = _tgt_dyn_transitions *
                      _current_targets.m_gaussians[i].mean;

    new_target.cov = _tgt_dyn_covariance +
                     _tgt_dyn_transitions *
                     _current_targets.m_gaussians[i].cov *
                     _tgt_dyn_transitions.transpose();

    // Push back to the expected targets
    _expected_targets.m_gaussians[i] = new_target;

    ++_n_predicted_targets;
  }
}

void GmphdFilter::print() {
  printf("Current gaussian mixture : \n");

  for (int i=0; i< _current_targets.m_gaussians.size(); ++i) {
    printf("Gaussian %d - pos %.1f  %.1f %.1f - cov %.1f  %.1f %.1f - weight %.3f\n",
           i,
           _current_targets.m_gaussians[i].mean(0,0),
           _current_targets.m_gaussians[i].mean(1,0),
           _current_targets.m_gaussians[i].mean(2,0),
           _current_targets.m_gaussians[i].cov(0,0),
           _current_targets.m_gaussians[i].cov(1,1),
           _current_targets.m_gaussians[i].cov(2,2),
           _current_targets.m_gaussians[i].weight) ;
  }
  printf("\n");
}

void  GmphdFilter::propagate () {
  _n_predicted_targets = 0;

  // Predict new targets (spawns):
  predictBirth();

  // Predict propagation of expected targets :
  predictTargets();

  // Build the update components
  buildUpdate ();

  if(_verbose) {
      printf("\nGMPHD_propagate :--- Expected targets : %d ---\n", _n_predicted_targets);
      _expected_targets.print();
  }

  // Update GMPHD
  update();
  if (_verbose) {
      printf("\nGMPHD_propagate :--- \n");
      _current_targets.print ();
  }

  // Prune gaussians (remove weakest, merge close enough gaussians)
  pruneGaussians ();

  if (_verbose) {
      printf("\nGMPHD_propagate :--- Pruned targets : ---\n");
      _current_targets.print();
  }

  // Clean vectors :
  _expected_measure.clear ();
  _expected_dispersion.clear ();
  _uncertainty.clear ();
  _covariance.clear ();
}

void  GmphdFilter::pruneGaussians() {

  GaussianMixture pruned_gaussians = _current_targets.prune (this->_prune_trunc_thld,
                                                             this->_prune_merge_thld,
                                                             this->_prune_max_nb);

  this->_current_targets = pruned_gaussians;
}

void GmphdFilter::reset() {
  this->_current_targets.m_gaussians.clear ();
  this->_extracted_targets.m_gaussians.clear ();
}


void  GmphdFilter::setBirthModel(vector<GaussianModel> &birth_model) {
  _birth_model.m_gaussians.clear ();
  _birth_model.m_gaussians = birth_model;
}

void  GmphdFilter::setDynamicsModel(float sampling,
                                    float process_noise) {

  _sampling_period  = sampling;
  _process_noise    = process_noise;

  // Fill in propagation matrix :
  _tgt_dyn_transitions = MatrixXf::Identity(_dim_state, _dim_state);

  for (int i = 0; i<_dim_measures; ++i) {
      _tgt_dyn_transitions(i,_dim_measures+i) = _sampling_period;
  }

  // Fill in covariance matrix
  // Extra covariance added by the dynamics. Could be 0.
  _tgt_dyn_covariance = process_noise * process_noise *
                        MatrixXf::Identity(_dim_state, _dim_state);

//  // FIXME: hardcoded crap !
//  _tgt_dyn_covariance(0,0) = powf (sampling, 4.f)/4.f;
//  _tgt_dyn_covariance(1,1) = powf (sampling, 4.f)/4.f;
//  _tgt_dyn_covariance(2,2) = powf (sampling, 4.f)/4.f;

//  _tgt_dyn_covariance(3,3) = powf (sampling, 2.f);
//  _tgt_dyn_covariance(4,4) = powf (sampling, 2.f);
//  _tgt_dyn_covariance(5,5) = powf (sampling, 2.f);

//  _tgt_dyn_covariance(0,3) = powf (sampling, 3.f)/2.f;
//  _tgt_dyn_covariance(1,4) = powf (sampling, 3.f)/2.f;
//  _tgt_dyn_covariance(2,5) = powf (sampling, 3.f)/2.f;

//  _tgt_dyn_covariance(3,0) = powf (sampling, 3.f)/2.f;
//  _tgt_dyn_covariance(4,1) = powf (sampling, 3.f)/2.f;
//  _tgt_dyn_covariance(5,1) = powf (sampling, 3.f)/2.f;

//  _tgt_dyn_covariance = _tgt_dyn_covariance *
//                       (_process_noise * _process_noise);
  // \FIXME
}

void GmphdFilter::setDynamicsModel(MatrixXf &tgt_dyn_transitions,
                                   MatrixXf &tgt_dyn_covariance) {

    _tgt_dyn_transitions = tgt_dyn_transitions;
    _tgt_dyn_covariance = tgt_dyn_covariance;
}

void  GmphdFilter::setNewMeasurements(vector<float> &position,
                                      vector<float> &speed) {

  // Clear the gaussian mixture
  this->_meas_targets.m_gaussians.clear();

  GaussianModel new_obs;

  for (int i=0; i< position.size()/3; ++i) {
    // Create new gaussian model according to measurement
    new_obs.mean(0,0) = position[3*i    ];
    new_obs.mean(1,0) = position[3*i + 1];
    new_obs.mean(2,0) = position[3*i + 2];

    new_obs.mean(3,0) = speed[3*i    ];
    new_obs.mean(4,0) = speed[3*i + 1];
    new_obs.mean(5,0) = speed[3*i + 2];

    // Covariance ?
    new_obs.cov = this->_obs_covariance;

    // Weight (?)
    new_obs.weight = 1.f;

    this->_meas_targets.m_gaussians.push_back(new_obs);
  }
}

void  GmphdFilter::setNewReferential(const Matrix4f *transform) {
  // Change referential for every gaussian in the gaussian mixture
  this->_current_targets.changeReferential(transform);
}



void  GmphdFilter::setPruningParameters (float  prune_trunc_thld,
                                         float  prune_merge_thld,
                                         int    prune_max_nb) {

  _prune_trunc_thld = prune_trunc_thld;
  _prune_merge_thld = prune_merge_thld;
  _prune_max_nb     = prune_max_nb;
}


void  GmphdFilter::setObservationModel(float prob_detection_overall,
                                       float measurement_noise_pose,
                                       float measurement_noise_speed,
                                       float measurement_background) {

  _p_detection_overall      = prob_detection_overall;
  _measurement_noise_pose   = measurement_noise_pose;
  _measurement_noise_speed  = measurement_noise_speed;
  _measurement_background   = measurement_background; // False detection probability

  // Set model matrices
  _obs_matrix      = MatrixXf::Identity(_dim_state, _dim_state);
  _obs_matrix_T = _obs_matrix.transpose();
  _obs_covariance  = MatrixXf::Identity(_dim_state,_dim_state);

  // FIXME: deal with the _motion_model parameter !
  _obs_covariance.block(0,0,_dim_measures, _dim_measures) *= _measurement_noise_pose * _measurement_noise_pose;
  _obs_covariance.block(_dim_measures,_dim_measures,_dim_state, _dim_state) *= _measurement_noise_speed * _measurement_noise_speed;
}

void  GmphdFilter::setSpawnModel(vector <SpawningModel, aligned_allocator<SpawningModel> > &_spawn_models) {
  // Stupid implementation, maybe to be improved..
  for (int i=0; i<_spawn_models.size(); ++i) {
    this->_spawn_model.push_back(_spawn_models[i]);
  }
}

void  GmphdFilter::setSurvivalProbability(float _prob_survival) {
  this->_p_survival_overall = _prob_survival;
}

void  GmphdFilter::update() {
  int n_meas, n_targt, index;
  _current_targets.m_gaussians.clear();

  // We'll consider every possible association : vector size is (expected targets)*(measured targets)
  _current_targets.m_gaussians.resize((_meas_targets.m_gaussians.size () + 1) *
                            _expected_targets.m_gaussians.size ());

  // First set of gaussians : mere propagation of existing ones
  // \warning : don't propagate the "birth" targets...
  // we set their weight to 0

  _n_predicted_targets =  _expected_targets.m_gaussians.size ();
  int i_birth_current = 0;
  for (int i=0; i<_n_predicted_targets; ++i) {
    if (i != _i_birth_targets[i_birth_current]) {
      _current_targets.m_gaussians[i].weight = (1.f - this->_p_detection_overall) *
                                    _expected_targets.m_gaussians[i].weight;
    } else {
      i_birth_current = min(i_birth_current+1, (int) _i_birth_targets.size ());
      _current_targets.m_gaussians[i].weight = 0.f;
    }

    _current_targets.m_gaussians[i].mean = _expected_targets.m_gaussians[i].mean;
    _current_targets.m_gaussians[i].cov  = _expected_targets.m_gaussians[i].cov;
  }


  // Second set of gaussians : match observations and previsions
  if (_meas_targets.m_gaussians.size () == 0) {
    return;
  } else {
    for (n_meas=1; n_meas <= _meas_targets.m_gaussians.size (); ++n_meas) {
      for (n_targt = 0; n_targt < _n_predicted_targets; ++n_targt) {

        index = n_meas * _n_predicted_targets + n_targt;

        // Compute matching factor between predictions and measures.
        // \warning : we only take positions into account there
        _current_targets.m_gaussians[index].weight =  _p_detection_overall *
                                            _expected_targets.m_gaussians[n_targt].weight *
                                            gaussDensity_3D(_meas_targets.m_gaussians[n_meas -1].mean.block(0,0,3,1),
                                                            _expected_measure[n_targt].block(0,0,3,1),
                                                            _expected_dispersion[n_targt].block(0,0,3,3));

        _current_targets.m_gaussians[index].mean =  _expected_targets.m_gaussians[n_targt].mean +
                                          _uncertainty[n_targt] *
                                          (_meas_targets.m_gaussians[n_meas -1].mean - _expected_measure[n_targt]);

        _current_targets.m_gaussians[index].cov = _covariance[n_targt];
      }

      // Normalize weights in the same predicted set,
      // taking clutter into account
      _current_targets.normalize (_measurement_background,
                                  n_meas       * _n_predicted_targets,
                                  (n_meas + 1) * _n_predicted_targets,
                                  1);
    }
  }
}
