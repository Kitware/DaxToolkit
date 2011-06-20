/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __DaxField_h
#define __DaxField_h

#include "DaxCommon.h"
#include "DaxWork.cu"

class DaxField
{
public:
};

class DaxFieldPoint : public DaxField
{
  SUPERCLASS(DaxField);
};

class DaxFieldCell : public DaxField
{
  SUPERCLASS(DaxField);
public:
  /// Set a scalar value.
  __device__ void Set(const DaxWorkMapCell& cell, DaxScalar scalar)
    {
    }
};


#endif
