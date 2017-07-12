#ifndef _QUDACXX_H
#define _QUDACXX_H

/**
 * @file  qudacxx.h
 * @brief C++ header file for the QUDA library
 *
 */

//#include <qudaconfigure.h>
#include <enum_quda.h>
#include <stdio.h> /* for FILE */
#include <quda_constants.h>
// include support for QUDA gauge_field, color_spinor_field and blas operations 
#include <gauge_field.h>
#include <blas_quda.h>

namespace quda{
void computeKSLinkQuda(cudaGaugeField* cudaFatLink, cudaGaugeField* cudaLongLink, cudaGaugeField* cudaUnitarizedLink, cudaGaugeField* cudaInLink, double *path_coeff, QudaGaugeParam *param);
}


#endif /* _QUDACXX_H */
