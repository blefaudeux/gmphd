#pragma once

// Author : Benjamin Lefaudeux (blefaudeux@github)

#include <Eigen/Eigen>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <Eigen/StdVector>
#include <math.h>

namespace gmphd {
using namespace std;
using namespace Eigen;

    template <size_t size>
    float pseudo_inv(Matrix<float, size, size> const &mat_in,
                    Matrix<float, size, size> &mat_out)
    {
        Matrix<float, size, size> U;
        Matrix<float, size, 1> eig_val;
        Matrix<float, size, size> eig_val_inv;
        Matrix<float, size, size> V;
        float det;

        eig_val_inv = MatrixXf::Identity(size, size);

        // Compute the SVD decomposition
        JacobiSVD<MatrixXf> svd(*mat_in,
                                            ComputeThinU | ComputeThinV);

        eig_val = svd.singularValues();
        U = svd.matrixU();
        V = svd.matrixV();

        // Compute pseudo-inverse
        // - quick'n'dirty inversion of eigen Matrix
        for (int i = 0; i < size; ++i)
        {
            if (eig_val(i, 0) != 0.f)
            {
                eig_val_inv(i, i) = 1.f / eig_val(i, 0);
            }
            else
            {
                eig_val_inv(i, i) = 0.f;
            }
        }

        *mat_out = V.transpose() * eig_val_inv * U.transpose();

        // Compute determinant from eigenvalues..
        det = 1.f;
        for (int i = 0; i < size; ++i)
        {
            det *= eig_val(i, 0);
        }

        return det;
    }

    template <size_t T>
    static float mahalanobis(const Matrix<float, T, 1> &point,
                            const Matrix<float, T, 1> &mean,
                            const Matrix<float, T, T> &cov)
    {
        int ps = point.rows();
        Matrix<float, T, 1> x_cen = point - mean;
        Matrix<float, T, T> b = Matrix<float, T, T>::Identity();

        // TODO: Ben - cov needs to be normalized !
        cov.ldlt().solveInPlace(b);
        x_cen = b * x_cen;
        return (x_cen.transpose() * x_cen).sum(); // FIXME / looks a bit fishy
    }

    template <size_t T>
    static float gaussDensity(const Matrix<float, T, 1> &point,
                            const Matrix<float, T, 1> &mean,
                            const Matrix<float, T, T> &cov)
    {
        float det, res;

        Matrix<float, T, T> cov_inverse;
        Matrix<float, T, 1> mismatch;

        det = cov.determinant();
        cov_inverse = cov.inverse();

        mismatch = point - mean;

        Matrix<float, 1, 1> distance = mismatch.transpose() * cov_inverse * mismatch;
        distance /= -2.f;

        // Deal with faulty determinant case
        if (det == 0.f)
        {
            return 0.f;
        }

        res = 1.f / sqrt(pow(2 * M_PI, T) * fabs(det)) * exp(distance.coeff(0, 0));

        if (isinf(det))
        {
            printf("Problem in multivariate gaussian\n distance : %f - det %f\n", distance.coeff(0, 0), det);
            cout << "Cov \n"
                << cov << endl
                << "Cov inverse \n"
                << cov_inverse << endl;
            return 0.f;
        }

        return res;
    }

}
