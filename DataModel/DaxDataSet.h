/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxDataSet_h
#define __DaxDataSet_h

#include "DaxDataArray.h"

#define MAX_NUMBER_OF_FIELDS 10

class DaxDataSet
{
public:
  int PointCoordinatesIndex;
  int CellArrayIndex;
  int CellDataIndices[MAX_NUMBER_OF_FIELDS];
  int PointDataIndices[MAX_NUMBER_OF_FIELDS];
  __device__ __host__ DaxDataSet() :
    PointCoordinatesIndex(-1),
    CellArrayIndex(-1)
  {
  for (int cc=0; cc < MAX_NUMBER_OF_FIELDS; cc++)
    {
    this->CellDataIndices[cc] = this->PointDataIndices[cc] = -1;
    }
  }
};

#endif
