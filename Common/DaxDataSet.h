/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxDataSet_h
#define __DaxDataSet_h

#include "DaxDataArray.h"

#define MAX_NUMBER_OF_FIELDS 10

/// DaxDataSet is the data-structure that encapsulates a data-set in the
/// execution environment. The user code never uses this class directly neither
/// worklets, nor in the control environment.
class DaxDataSet
{
public:
  int PointCoordinatesIndex;
  int CellArrayIndex;
  int CellDataIndices[MAX_NUMBER_OF_FIELDS];
  int PointDataIndices[MAX_NUMBER_OF_FIELDS];
  DaxDataSet() :
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
