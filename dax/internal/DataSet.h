/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_core_DataSet_h
#define __dax_core_DataSet_h

#include <dax/internal/DataArray.h>

#define MAX_NUMBER_OF_FIELDS 10

namespace dax { namespace internal {

/// DataSet is the data-structure that encapsulates a data-set in the
/// execution environment. The user code never uses this class directly neither
/// worklets, nor in the control environment.
class DataSet
{
public:
  int PointCoordinatesIndex;
  int CellArrayIndex;
  int CellDataIndices[MAX_NUMBER_OF_FIELDS];
  int PointDataIndices[MAX_NUMBER_OF_FIELDS];
  DAX_EXEC_CONT_EXPORT DataSet() :
    PointCoordinatesIndex(-1),
    CellArrayIndex(-1)
  {
  for (int cc=0; cc < MAX_NUMBER_OF_FIELDS; cc++)
    {
    this->CellDataIndices[cc] = this->PointDataIndices[cc] = -1;
    }
  }
};

}}

#endif
