/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxDataSet_h
#define __DaxDataSet_h

#include "DaxDataArray.h"

class DaxDataSet
{
public:
  int PointCoordinatesIndex;
  int CellArrayIndex;
  int *CellDataIndices;
  int *PointDataIndices;
  __device__ __host__ DaxDataSet() :
    PointCoordinatesIndex(-1),
    CellArrayIndex(-1),
    CellDataIndices(NULL),
    PointDataIndices(NULL)
  {
  }
};

#endif
