#ifndef __dax_cont_Modules_h
#define __dax_cont_Modules_h

#include <cuda_runtime.h>
#include <boost/throw_exception.hpp>
#include <algorithm>

#include <dax/Types.h>

namespace dax { namespace cuda { namespace control { namespace internal {

struct exception_base: virtual std::exception, virtual boost::exception { };
struct no_cuda_device: virtual exception_base { };

struct CudaInfo
{
  int NumThreadsPerBlock;
  int MaxGridSize;
};

/// Holds the basic information needed to determine the number of threads and
/// blocks to run on the current cuda card. The first time this class is
/// constructed it fills a static struct of information queired from the
/// device.
///
class CudaParameters
{
public:
  /// Construct a CudaParameters using a grid (or grid-like) structure.
  /// The class is expected to have const methods named GetNumberOfPoints
  /// and GetNumberOfCells.
  ///
  template< typename T>
  CudaParameters(const T& source)
  {
    dax::Id numPoints = source.GetNumberOfPoints();
    dax::Id numCells = source.GetNumberOfCells();

    this->InitializeParameters(numPoints, numCells);
  }

  /// Constructs a CudaParameters using an unspecified grid of the given number
  /// of points and cells.
  ///
  CudaParameters(dax::Id numPoints, dax::Id numCells)
  {
    this->InitializeParameters(numPoints, numCells);
  }

  dax::Id GetNumberOfPointBlocks() const { return NumPointBlocks; }
  dax::Id GetNumberOfPointThreads() const { return NumPointThreads; }

  dax::Id GetNumberOfCellBlocks() const { return NumCellBlocks; }
  dax::Id GetNumberOfCellThreads() const { return NumCellThreads; }
private:
  dax::Id NumPointBlocks;
  dax::Id NumPointThreads;
  dax::Id NumCellBlocks;
  dax::Id NumCellThreads;

  /// Set the internal fields.
  void InitializeParameters(dax::Id numPoints, dax::Id numCells)
  {
    const CudaInfo &cardInfo = this->QueryDevice();

    this->NumPointThreads = cardInfo.NumThreadsPerBlock;
    this->NumCellThreads = cardInfo.NumThreadsPerBlock;

    //determine the max number of blocks that we can have
    this->NumPointBlocks
        = (numPoints+this->NumPointThreads-1)/this->NumPointThreads;
    this->NumCellBlocks
        = (numCells+this->NumPointThreads-1)/this->NumPointThreads;

    //make sure we don't request too many blocks for the card
    this->NumPointBlocks = std::min(cardInfo.MaxGridSize, this->NumPointBlocks);
    this->NumCellBlocks = std::min(cardInfo.MaxGridSize, this->NumCellBlocks);
  }

  /// Queries the machine for cuda devices. Currently always selects the first
  /// device as the one to use for computation.
  ///
  const CudaInfo &QueryDevice()
    {
    static bool cardQueried = false;
    static CudaInfo cardInfo;
    if(!cardQueried)
      {
      int devCount=0;
      cudaGetDeviceCount(&devCount);
      if (devCount<=0)
        {
        //can't run the code if we don't have a device
        BOOST_THROW_EXCEPTION(no_cuda_device());
        }

      cudaDeviceProp devProp;
      cudaGetDeviceProperties(&devProp, 0);

      //get the numThreadsPerBlock
      //we want 128 threads but will take what we can get
      cardInfo.NumThreadsPerBlock =
          std::min(devProp.maxThreadsPerBlock,128);

      //figure out the max grid size
      cardInfo.MaxGridSize = devProp.maxGridSize[0];

      cardQueried = true;
      }
    return cardInfo;
    }
};

}}}}

#endif
