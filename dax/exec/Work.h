/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
/// This file defines classes identifying "work" done by different kinds of
/// worklets.
#ifndef __dax_exec_Work_h
#define __dax_exec_Work_h

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/exec/internal/Macros.h>

namespace dax {  namespace exec {

///----------------------------------------------------------------------------
/// Base-class for all different types of work.
class Work
{
protected:
  dax::Id Item;
  dax::Id Iteration;

  __device__ Work(dax::Id cur_iteration) : Iteration(cur_iteration)
    {
#ifdef __CUDACC__
    this->Item = (dax::Id) (blockIdx.x * blockDim.x + threadIdx.x
      + Iteration * gridDim.x * blockDim.x);
#else
    this->Item = 0;
#endif
    }
public:
  __device__ dax::Id GetItem() const { return this->Item; }
  __device__ void SetItem(dax::Id id)
    {
    this->Item = id;
    }
};

///----------------------------------------------------------------------------
/// Work for worklets that map fields without regard to topology or any other
/// connectivity information
class WorkMapField : public dax::exec::Work
{
  SUPERCLASS(dax::exec::Work);
public:
  __device__ WorkMapField(dax::Id cur_iteration=0): Superclass(cur_iteration)
    {

    }
};

///----------------------------------------------------------------------------
// Work for worklets that map points to cell. Use this work when the worklets
// need "CellArray" information i.e. information about what points form a cell.
class WorkMapCell : public dax::exec::WorkMapField
{
  SUPERCLASS(dax::exec::WorkMapField);
  const dax::internal::DataArray& CellArray;
public:

  __device__ const dax::internal::DataArray& GetCellArray() const
    {
    return this->CellArray;
    }

  __device__ WorkMapCell(const dax::internal::DataArray& cell_array,
    dax::Id cur_iteration) :
    Superclass(cur_iteration), CellArray(cell_array) 
    {
    }
};

}}
#endif
