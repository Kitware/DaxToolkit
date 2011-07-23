/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
/// This file defines classes identifying "work" done by different kinds of
/// worklets.
#ifndef __DaxWork_h
#define __DaxWork_h

#include "DaxCommon.h"

class DaxDataArray;

///----------------------------------------------------------------------------
/// Base-class for all different types of work.
class DaxWork
{
protected:
  DaxId Item;

  __device__ DaxWork()
    {
    this->Item = (DaxId) (blockIdx.x * blockDim.x + threadIdx.x);
    }
public:
  __device__ DaxId GetItem() const { return this->Item; }
  __device__ void SetItem(DaxId id)
    {
    this->Item = id;
    }

};

///----------------------------------------------------------------------------
/// Work for worklets that map fields without regard to topology or any other
/// connectivity information
class DaxWorkMapField : public DaxWork
{
  SUPERCLASS(DaxWork);
public:
  __device__ DaxWorkMapField()
    {

    }
};

///----------------------------------------------------------------------------
// Work for worklets that map points to cell. Use this work when the worklets
// need "CellArray" information i.e. information about what points form a cell.
class DaxWorkMapCell : public DaxWorkMapField
{
  SUPERCLASS(DaxWorkMapField);
  const DaxDataArray& CellArray;
public:

  __device__ const DaxDataArray& GetCellArray() const
    {
    return this->CellArray;
    }

  __device__ DaxWorkMapCell(const DaxDataArray& cell_array) : CellArray(cell_array)
    {
    }
};

#endif
