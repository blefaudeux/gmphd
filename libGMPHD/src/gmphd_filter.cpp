#include "gmphd_filter.h"


// Author : Benjamin Lefaudeux (blefaudeux@github)


GMPHD::GMPHD(int max_gaussians,
             int dimension,
             bool motion_model,
             bool verbose):m_maxGaussians(max_gaussians),
    m_dimMeasures(dimension),
    m_motionModel(motion_model),
    m_bVerbose(verbose){

    if (!motion_model)
        m_dimState = m_dimMeasures;
    else
        m_dimState = 2 * m_dimMeasures;

    // Initiate matrices :
    I = MatrixXf::Identity(m_dimState, m_dimState);
}

void  GMPHD::buildUpdate ()
{
    MatrixXf temp_matrix(m_dimState, m_dimState);

    // Concatenate all the wannabe targets :
    // - birth targets
    m_iBirthTargets.clear();

    if(_birth_targets.m_gaussians.size () > 0)
    {
        for (int i=0; i<_birth_targets.m_gaussians.size (); ++i)
        {
            m_iBirthTargets.push_back (_expected_targets.m_gaussians.size () + i);
        }

        _expected_targets.m_gaussians.insert (_expected_targets.m_gaussians.end (),
                                              _birth_targets.m_gaussians.begin (),
                                              _birth_targets.m_gaussians.begin () + _birth_targets.m_gaussians.size ());
    }

    // - spawned targets
    if (_spawned_targets.m_gaussians.size () > 0)
    {
        _expected_targets.m_gaussians.insert (_expected_targets.m_gaussians.end (),
                                              _spawned_targets.m_gaussians.begin (),
                                              _spawned_targets.m_gaussians.begin () + _spawned_targets.m_gaussians.size ());
    }

    if (m_bVerbose)
    {
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
    m_nPredictedTargets = _expected_targets.m_gaussians.size ();

    _expected_measure.resize (m_nPredictedTargets);
    _expected_dispersion.resize (m_nPredictedTargets);
    _uncertainty.resize (m_nPredictedTargets);
    _covariance.resize (m_nPredictedTargets);

    for (int i=0; i< m_nPredictedTargets; ++i)
    {
        // Compute the expected measurement
        _expected_measure[i] = _obs_matrix * _expected_targets.m_gaussians[i].m_mean;

        _expected_dispersion[i] = _obs_covariance + _obs_matrix * _expected_targets.m_gaussians[i].m_cov * _obs_matrix_T;

        if (isnan(_expected_dispersion[i](0,0)))
        {
            printf("NaN value in dispersion\n");
            cout << "Expected cov \n" << _expected_targets.m_gaussians[i].m_cov << endl << endl;
            THROW_ERR("NaN in GMPHD Update process");
        }

        temp_matrix = _expected_dispersion[i].inverse();

        _uncertainty[i] =  _expected_targets.m_gaussians[i].m_cov * _obs_matrix_T * temp_matrix;

        _covariance[i] = (I - _uncertainty[i]*_obs_matrix) * _expected_targets.m_gaussians[i].m_cov;
    }
}

void    GMPHD::extractTargets(float threshold)
{
    // Deal with crappy settings
    float thld = max(threshold, 0.f);

    // Get trough every target, keep the ones whose weight is above threshold
    _extracted_targets.m_gaussians.clear();

    for (int i=0; i<_current_targets.m_gaussians.size(); ++i)
    {
        if (_current_targets.m_gaussians[i].m_weight >= thld)
        {
            _extracted_targets.m_gaussians.push_back(_current_targets.m_gaussians[i]);
        }
    }

    printf("GMPHD_extract : %d targets\n", _extracted_targets.m_gaussians.size ());
}

void    GMPHD::getTrackedTargets(const float extract_thld, vector<float> &position,
                                 vector<float> &speed, vector<float> &weight)
{

    // Fill in "extracted_targets" from the "current_targets"
    extractTargets(extract_thld);

    position.clear();
    speed.clear();
    weight.clear();

    for (int i=0; i< _extracted_targets.m_gaussians.size(); ++i)
    {
        position.push_back(_extracted_targets.m_gaussians[i].m_mean(0,0));
        position.push_back(_extracted_targets.m_gaussians[i].m_mean(1,0));
        position.push_back(_extracted_targets.m_gaussians[i].m_mean(2,0));

        speed.push_back(_extracted_targets.m_gaussians[i].m_mean(3,0));
        speed.push_back(_extracted_targets.m_gaussians[i].m_mean(4,0));
        speed.push_back(_extracted_targets.m_gaussians[i].m_mean(5,0));

        weight.push_back(_extracted_targets.m_gaussians[i].m_weight);
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
float   GMPHD::gaussDensity(MatrixXf const & point, MatrixXf const & mean, MatrixXf const & cov)
{
    MatrixXf cov_inverse, mismatch;

    float det, res, dist;

    det         = cov.block(0, 0, m_dimMeasures, m_dimMeasures).determinant();
    cov_inverse = cov.block(0, 0, m_dimMeasures, m_dimMeasures).inverse();
    mismatch    = point.block(0,0,m_dimMeasures, 1) - mean.block(0,0,m_dimMeasures,1);

    Matrix <float,1,1> distance = mismatch.transpose() * cov_inverse * mismatch;

    distance /= -2.f;

    dist =  (float) distance(0,0);
    res = 1.f/(pow(2*M_PI, m_dimMeasures) * sqrt(fabs(det)) * exp(dist));

    return res;
}

float   GMPHD::gaussDensity_3D(const Matrix <float, 3,1> &point,
                               const Matrix <float, 3,1> &mean,
                               const Matrix <float, 3,3> &cov)
{
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
    {
        return 0.f;
    }

    res = 1.f/sqrt(pow(2*M_PI, 3) * fabs(det)) * exp(distance.coeff (0,0));

    if (isinf(det))
    {
        printf("Problem in multivariate gaussian\n distance : %f - det %f\n", distance.coeff (0,0), det);
        cout << "Cov \n" << cov << endl << "Cov inverse \n" << cov_inverse << endl;
        return 0.f;
    }

    return res;
}

void  GMPHD::predictBirth()
{
    _spawned_targets.m_gaussians.clear ();
    _birth_targets.m_gaussians.clear ();

    // -----------------------------------------
    // Compute spontaneous births
    _birth_targets.m_gaussians = _birth_model.m_gaussians;
    m_nPredictedTargets += _birth_targets.m_gaussians.size ();

    // -----------------------------------------
    // Compute spawned targets
    GaussianModel new_spawn;

    for (int k=0; k< _current_targets.m_gaussians.size (); ++k)
    {
        for (int i=0; i< _spawn_model.size (); ++i)
        {
            // Define a gaussian model from the existing target
            // and spawning properties
            new_spawn.m_weight = _current_targets.m_gaussians[k].m_weight
                               * _spawn_model[i].m_weight;

            new_spawn.m_mean = _spawn_model[i].m_offset
                             + _spawn_model[i].m_trans * _current_targets.m_gaussians[k].m_mean;

            new_spawn.m_cov = _spawn_model[i].m_cov
                            + _spawn_model[i].m_trans *
                            _current_targets.m_gaussians[k].m_cov *
                            _spawn_model[i].m_trans.transpose();

            // Add this new gaussian to the list of expected targets
            _spawned_targets.m_gaussians.push_back (new_spawn);

            // Update the number of expected targets
            m_nPredictedTargets ++;
        }
    }
}

void  GMPHD::predictTargets () {
    GaussianModel new_target;

    _expected_targets.m_gaussians.resize(_current_targets.m_gaussians.size ());

    for (int i=0; i<_current_targets.m_gaussians.size (); ++i)
    {
        // Compute the new shape of the target
        new_target.m_weight = m_pSurvival * _current_targets.m_gaussians[i].m_weight;

        new_target.m_mean = _tgt_dyn_transitions * _current_targets.m_gaussians[i].m_mean;

        new_target.m_cov = _tgt_dyn_covariance +
                         _tgt_dyn_transitions * _current_targets.m_gaussians[i].m_cov * _tgt_dyn_transitions.transpose();

        // Push back to the expected targets
        _expected_targets.m_gaussians[i] = new_target;

        ++m_nPredictedTargets;
    }
}

void GMPHD::print()
{
    printf("Current gaussian mixture : \n");

    for (int i=0; i< _current_targets.m_gaussians.size(); ++i)
    {
        printf("Gaussian %d - pos %.1f  %.1f %.1f - cov %.1f  %.1f %.1f - weight %.3f\n",
               i,
               _current_targets.m_gaussians[i].m_mean(0,0),
               _current_targets.m_gaussians[i].m_mean(1,0),
               _current_targets.m_gaussians[i].m_mean(2,0),
               _current_targets.m_gaussians[i].m_cov(0,0),
               _current_targets.m_gaussians[i].m_cov(1,1),
               _current_targets.m_gaussians[i].m_cov(2,2),
               _current_targets.m_gaussians[i].m_weight) ;
    }
    printf("\n");
}

void  GMPHD::propagate ()
{
    m_nPredictedTargets = 0;

    // Predict new targets (spawns):
    predictBirth();

    // Predict propagation of expected targets :
    predictTargets();

    // Build the update components
    buildUpdate ();

    if(m_bVerbose)
    {
        printf("\nGMPHD_propagate :--- Expected targets : %d ---\n", m_nPredictedTargets);
        _expected_targets.print();
    }

    // Update GMPHD
    update();

    if (m_bVerbose)
    {
        printf("\nGMPHD_propagate :--- \n");
        _current_targets.print ();
    }

    // Prune gaussians (remove weakest, merge close enough gaussians)
    pruneGaussians ();

    if (m_bVerbose)
    {
        printf("\nGMPHD_propagate :--- Pruned targets : ---\n");
        _current_targets.print();
    }

    // Clean vectors :
    _expected_measure.clear ();
    _expected_dispersion.clear ();
    _uncertainty.clear ();
    _covariance.clear ();
}

void  GMPHD::pruneGaussians()
{
    _current_targets.prune( m_pruneTruncThld, m_pruneMergeThld, m_nMaxPrune );
}

void GMPHD::reset()
{
    _current_targets.m_gaussians.clear ();
    _extracted_targets.m_gaussians.clear ();
}


void  GMPHD::setBirthModel(vector<GaussianModel> &birth_model)
{
    _birth_model.m_gaussians.clear ();
    _birth_model.m_gaussians = birth_model;
}

void  GMPHD::setDynamicsModel(float sampling, float process_noise)
{

    m_samplingPeriod  = sampling;
    m_processNoise    = process_noise;

    // Fill in propagation matrix :
    _tgt_dyn_transitions = MatrixXf::Identity(m_dimState, m_dimState);

    for (int i = 0; i<m_dimMeasures; ++i)
    {
        _tgt_dyn_transitions(i,m_dimMeasures+i) = m_samplingPeriod;
    }

    // Fill in covariance matrix
    // Extra covariance added by the dynamics. Could be 0.
    _tgt_dyn_covariance = process_noise * process_noise *
                          MatrixXf::Identity(m_dimState, m_dimState);

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

void GMPHD::setDynamicsModel(MatrixXf &tgt_dyn_transitions, MatrixXf &tgt_dyn_covariance)
{
    _tgt_dyn_transitions = tgt_dyn_transitions;
    _tgt_dyn_covariance = tgt_dyn_covariance;
}

void  GMPHD::setNewMeasurements(vector<float> &position, vector<float> &speed)
{
    // Clear the gaussian mixture
    _meas_targets.m_gaussians.clear();

    GaussianModel new_obs;

    for (int i=0; i< position.size()/3; ++i) {
        // Create new gaussian model according to measurement
        new_obs.m_mean(0,0) = position[3*i    ];
        new_obs.m_mean(1,0) = position[3*i + 1];
        new_obs.m_mean(2,0) = position[3*i + 2];

        new_obs.m_mean(3,0) = speed[3*i    ];
        new_obs.m_mean(4,0) = speed[3*i + 1];
        new_obs.m_mean(5,0) = speed[3*i + 2];

        // Covariance ?
        new_obs.m_cov = _obs_covariance;

        // Weight (?)
        new_obs.m_weight = 1.f;

        _meas_targets.m_gaussians.push_back(new_obs);
    }
}

void  GMPHD::setNewReferential(const Matrix4f *transform)
{
    // Change referential for every gaussian in the gaussian mixture
    _current_targets.changeReferential(transform);
}



void  GMPHD::setPruningParameters (float  prune_trunc_thld,
                                   float  prune_merge_thld,
                                   int    prune_max_nb)
{

    m_pruneTruncThld = prune_trunc_thld;
    m_pruneMergeThld = prune_merge_thld;
    m_nMaxPrune     = prune_max_nb;
}


void  GMPHD::setObservationModel(float prob_detection_overall,
                                 float measurement_noise_pose,
                                 float measurement_noise_speed,
                                 float measurement_background)
{
    m_pDetection      = prob_detection_overall;
    m_measNoisePose   = measurement_noise_pose;
    m_measNoiseSpeed  = measurement_noise_speed;
    m_measNoiseBackground   = measurement_background; // False detection probability

    // Set model matrices
    _obs_matrix      = MatrixXf::Identity(m_dimState, m_dimState);
    _obs_matrix_T = _obs_matrix.transpose();
    _obs_covariance  = MatrixXf::Identity(m_dimState,m_dimState);

    // FIXME: deal with the _motion_model parameter !
    _obs_covariance.block(0,0,m_dimMeasures, m_dimMeasures) *= m_measNoisePose * m_measNoisePose;
    _obs_covariance.block(m_dimMeasures,m_dimMeasures,m_dimState, m_dimState) *= m_measNoiseSpeed * m_measNoiseSpeed;
}

void  GMPHD::setSpawnModel(vector <SpawningModel, aligned_allocator<SpawningModel> > &_spawn_models)
{
    // Stupid implementation, maybe to be improved..
    for (int i=0; i<_spawn_models.size(); ++i) {
        _spawn_model.push_back(_spawn_models[i]);
    }
}

void  GMPHD::setSurvivalProbability(float _prob_survival)
{
    m_pSurvival = _prob_survival;
}

void  GMPHD::update()
{
    int n_meas, n_targt, index;
    _current_targets.m_gaussians.clear();

    // We'll consider every possible association : vector size is (expected targets)*(measured targets)
    _current_targets.m_gaussians.resize((_meas_targets.m_gaussians.size () + 1) *
                                        _expected_targets.m_gaussians.size ());

    // First set of gaussians : mere propagation of existing ones
    // \warning : don't propagate the "birth" targets...
    // we set their weight to 0

    m_nPredictedTargets =  _expected_targets.m_gaussians.size ();
    int i_birth_current = 0;
    for (int i=0; i<m_nPredictedTargets; ++i) {
        if (i != m_iBirthTargets[i_birth_current]) {
            _current_targets.m_gaussians[i].m_weight = (1.f - m_pDetection) *
                                                     _expected_targets.m_gaussians[i].m_weight;
        } else {
            i_birth_current = min(i_birth_current+1, (int) m_iBirthTargets.size ());
            _current_targets.m_gaussians[i].m_weight = 0.f;
        }

        _current_targets.m_gaussians[i].m_mean = _expected_targets.m_gaussians[i].m_mean;
        _current_targets.m_gaussians[i].m_cov  = _expected_targets.m_gaussians[i].m_cov;
    }


    // Second set of gaussians : match observations and previsions
    if (_meas_targets.m_gaussians.size () == 0)
    {
        return;
    }
    else
    {
        for (n_meas=1; n_meas <= _meas_targets.m_gaussians.size (); ++n_meas)
        {
            for (n_targt = 0; n_targt < m_nPredictedTargets; ++n_targt)
            {
                index = n_meas * m_nPredictedTargets + n_targt;

                // Compute matching factor between predictions and measures.
                // \warning : we only take positions into account there
                _current_targets.m_gaussians[index].m_weight =  m_pDetection *
                                                              _expected_targets.m_gaussians[n_targt].m_weight *
                                                              gaussDensity_3D(_meas_targets.m_gaussians[n_meas -1].m_mean.block(0,0,3,1),
                        _expected_measure[n_targt].block(0,0,3,1),
                        _expected_dispersion[n_targt].block(0,0,3,3));

                _current_targets.m_gaussians[index].m_mean =  _expected_targets.m_gaussians[n_targt].m_mean +
                                                            _uncertainty[n_targt] *
                                                            (_meas_targets.m_gaussians[n_meas -1].m_mean - _expected_measure[n_targt]);

                _current_targets.m_gaussians[index].m_cov = _covariance[n_targt];
            }

            // Normalize weights in the same predicted set,
            // taking clutter into account
            _current_targets.normalize (m_measNoiseBackground, n_meas * m_nPredictedTargets,
                                        (n_meas + 1) * m_nPredictedTargets, 1);
        }
    }
}
