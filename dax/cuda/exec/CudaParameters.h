#ifndef __dax_cont_Modules_h
#define __dax_cont_Modules_h

#include <dax/Types.h>
#include <dax/internal/GridStructures.h>

namespace dax { namespace cuda { namespace exec {
class CudaParameters
{
  //holds the basic information needed to determine the
  //number of threads and blocks to run on the current cuda card
public:
  template< typename T>
  CudaParameters(T source):
    NumPointThreads(128),
    NumCellThreads(128)
  {
    dax::Id numPts = dax::internal::numberOfPoints(source);
    dax::Id numCells = dax::internal::numberOfCells(source);

    NumPointBlocks = (numPts+NumPointThreads-1)/NumPointThreads;
    NumCellBlocks = (numCells+NumPointThreads-1)/NumPointThreads;
  }

  dax::Id numPointBlocks() const { return NumPointBlocks; }
  dax::Id numPointThreads() const { return NumPointThreads; }

  dax::Id numCellBlocks() const { return NumCellBlocks; }
  dax::Id numCellThreads() const { return NumCellThreads; }
protected:
  dax::Id NumPointBlocks;
  dax::Id NumPointThreads;
  dax::Id NumCellBlocks;
  dax::Id NumCellThreads;
};
}}}

#endif
