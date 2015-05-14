#ifndef GAUSSIAN_MIXTURE_H
#define GAUSSIAN_MIXTURE_H

// Author : Benjamin Lefaudeux (blefaudeux@github)


#include "eigen_tools.h"
#include <list>
#include <algorithm>

using namespace std;
using namespace Eigen;

/*!
 * \brief Stupid index structure to help sort gaussian mixtures
 */
struct index_w {
    float weight;
    int   index;
};

/*!
 * \brief The gaussian_model struct for 3D targets & measurements -> 6D dimension
 */
struct GaussianModel {
  Matrix<float, 6,1, DontAlign> mean;
  Matrix<float, 6,6, DontAlign> cov;

  float weight;

  void reset();
};

/*!
 * \brief The gaussian_mixture is a sum of gaussian models,
 *  with according weights. Everything is public, no need to get/set...
 */
class GaussianMixture {
  public :
    /*!
     * \brief Default constructor
     */
    GaussianMixture();

    /*!
     * \brief Copy constructor (used in vector stack)
     * \param source
     */
    GaussianMixture(const GaussianMixture &source);

    /*!
     * \brief Assignment operator
     * \param source
     * \return
     */
    GaussianMixture operator=(const GaussianMixture &source);


    /*!
     * \brief Merge gaussians in the vector input,
     * remove initial gaussians from the gaussian mixture
     * and add the merged on
     *
     * \param i_gaussians_to_merge :
     * indexes of the gaussians from the
     * gaussian mixture to be merged (and removed from the mixture)
     *
     * \param b_remove_from_mixture:
     * if the gaussians to merge must be removed from the mixture
     *
     * \param destination :
     * (output) the merged gaussian
     *
     */
    GaussianModel mergeGaussians(vector<int> &i_gaussians_to_merge,
                                 bool b_remove_from_mixture);


    /*!
     * \brief Normalize the gaussian mixture :
     *  vec(i).weight << vec(i).weight / (linear_offset + sum(vec[].weight))
     *
     * \param linear_offset :
     *  extra offset to normalization. 0 if geometric mean
     */
    void  normalize(float linear_offset);

    /*!
     * \brief Normalize the gaussian mixture between start and stop :
     *  vec(i).weight << vec(i).weight / (linear_offset + sum(vec[].weight))
     *
     * \param linear_offset :
     *  extra offset to normalization. 0 if geometric mean
     *
     * \param start_pos
     * \param stop_pos
     */
    void  normalize(float linear_offset,
                    int   start_pos,
                    int   stop_pos,
                    int   step);

    /*!
     * \brief print all the Gaussians on stdout
     */
    void print();

    /*!
     * \brief prune           : prune all the gaussians from a gaussian mixture
     * \param trunc_threshold : threshold to get rid of a weak gaussian
     * \param merge_threshold : distance threshold to merge two gaussians
     * \param max_gaussians   : max number of gaussians in the end
     * \return the pruned gaussian mixture
     */
    GaussianMixture  prune(float  trunc_threshold,
                           float  merge_threshold,
                           int    max_gaussians);

    /*!
     * \brief Sort gaussians in the gaussian mixture by weight order
     */
    void qsort();

    /*!
     * \brief Select all the gaussians close (up to the threshold) to a given specimen
     * \param ref_gaussian_index
     * \param threshold
     * \param close_gaussians
     */
    void selectCloseGaussians(int         i_ref,
                              float       threshold,
                              vector<int> &close_gaussians);

    /*!
     * \brief Select the gaussian whose weight is the bigger from the gaussian mixture
     * \return
     */
    int selectBestGaussian();

    /*!
     * \brief changeReferential : change the referential of the gaussian model
     * \param trans_mat
     */
    void changeReferential(const Matrix4f *tranform);

 public:
    vector <GaussianModel> m_gaussians;


};

#endif // GAUSSIAN_MIXTURE_H
