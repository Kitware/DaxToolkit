#ifndef __dax_cont_Modules_h
#define __dax_cont_Modules_h

#include <cuda_runtime.h>
#include <boost/throw_exception.hpp>
#include <algorithm>

#include <dax/Types.h>
#include <dax/internal/GridStructures.h>

namespace dax { namespace cuda { namespace control { namespace internal {

struct exception_base: virtual std::exception, virtual boost::exception { };
struct no_cuda_device: virtual exception_base { };

struct cudaInfo
{
  int numThreadsPerBlock;
  int maxGridSize;
};

class CudaParameters
{
  //holds the basic information needed to determine the
  //number of threads and blocks to run on the current cuda card
  //the first time this class is constructed it fills a static
  //struct of information queired from the device.
public:
  template< typename T>
  CudaParameters(const T& source)
  {
    if(!CardQueryed)
      {
      queryDevice();
      }
    NumPointThreads = this->CardInfo.numThreadsPerBlock;
    NumCellThreads = this->CardInfo.numThreadsPerBlock;

    dax::Id numPts = source.numPoints();
    dax::Id numCells = source.numCells();

    //determine the max number of blocks that we can have
    NumPointBlocks = (numPts+NumPointThreads-1)/NumPointThreads;
    NumCellBlocks = (numCells+NumPointThreads-1)/NumPointThreads;

    //make sure we don't request too many blocks for the card
    NumPointBlocks = std::min(this->CardInfo.maxGridSize,NumPointBlocks);
    NumCellBlocks = std::min(this->CardInfo.maxGridSize,NumCellBlocks);
  }

  //queries the machine for cuda devices
  //currently always selects the first device as the one
  //to use for computation
  void queryDevice()
    {
    if(!CardQueryed)
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
      CudaParameters::CardInfo.numThreadsPerBlock =
          std::min(devProp.maxThreadsPerBlock,128);

      //figure out the max grid size
      CudaParameters::CardInfo.maxGridSize = devProp.maxGridSize[0];
      }
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

  static bool CardQueryed;
  static cudaInfo CardInfo;
};

bool CudaParameters::CardQueryed = false;
cudaInfo CudaParameters::CardInfo;
}}}}

#endif
