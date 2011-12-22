#ifndef __dax_cuda_cont_Worklets_h
#define __dax_cuda_cont_Worklets_h


#include <cuda.h>

#include <dax/Types.h>

#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>

#include <dax/cont/StructuredGrid.h>
#include <dax/cont/internal/ConvertTypes.h>
#include <dax/cont/internal/ArrayContainer.h>

#include <dax/cuda/cont/internal/DeviceArray.h>
#include <dax/cuda/cont/internal/WorkletFunctions.h>
#include <dax/cuda/exec/CudaParameters.h>


namespace dax { namespace cuda {  namespace cont { namespace worklet {

namespace
{
  template<typename T>
  dax::cuda::cont::internal::DeviceArrayPtr<T>
  locateDeviceArray(dax::cont::internal::ArrayContainer<T>& container)
    {
    //get device arrays if they already exists, otherwise
    //convert the control arrays to device arrays

    //need the template keyword to help out some compilers
    //figure out that execution is a templated method
    return container.template execution<
        dax::cuda::cont::internal::DeviceArray<T> >();
    }

}

class Elevation
{
public:
  template<typename G, typename T, typename U>
  void operator()(G &g,
                  dax::cont::internal::ArrayContainer<T>& in,
                  U& out)
  {
    typedef typename U::ValueType OutType;

    //get the size of the field we are iterating over
    //this is derived by the output array field type (points, cells)
    dax::Id size = U::fieldSize(g);

    dax::cuda::cont::internal::DeviceArrayPtr<T> ind(locateDeviceArray(in));
    //the reference counting is not working properly so we have to set ind again


    dax::cuda::cont::internal::DeviceArrayPtr<OutType> outd(
          new dax::cuda::cont::internal::DeviceArray<OutType>(size));

    assert(ind->size()==outd->size());

    //determine the cuda parameters from the data structure
    dax::cuda::exec::CudaParameters params(g);

    //this type needs to be auto derived
    dax::internal::StructureUniformGrid grid = dax::cont::internal::convert(g);

    //you can't use implicit constructors when calling a global
    //function
    dax::internal::DataArray<T> inField(ind);
    dax::internal::DataArray<OutType> outField(outd);

    std::cout << "Elevation" << std::endl;
    dax::cuda::cont::internal::ElevationFunction<<<params.numPointBlocks(),
        params.numPointThreads()>>>(size,
                                    grid,
                                    inField,
                                    outField);

    //associate the resulting device array with this host array
    //the array will be pulled of the device when the user requests
    //it on the client. Otherwise it will stay on the device
    out.setExecutionArray(outd);
  }
};

class Square
{
public:
  template<typename G, typename T, typename U>
  void operator()(G &g,
                  dax::cont::internal::ArrayContainer<T>& in,
                  U& out)
  {
    typedef typename U::ValueType OutType;

    //get the size of the field we are iterating over
    //this is derived by the output array field type (points, cells)
    dax::Id size = U::fieldSize(g);

    dax::cuda::cont::internal::DeviceArrayPtr<T> ind(locateDeviceArray(in));
    dax::cuda::cont::internal::DeviceArrayPtr<OutType> outd(
          new dax::cuda::cont::internal::DeviceArray<OutType>(size));

    assert(ind->size()==outd->size());

    //determine the cuda parameters from the data structure
    dax::cuda::exec::CudaParameters params(g);

    //this type needs to be auto derived
    dax::internal::StructureUniformGrid grid = dax::cont::internal::convert(g);

    //this should be easier to do
    dax::internal::DataArray<T> inField(ind);
    dax::internal::DataArray<OutType> outField(outd);

    std::cout << "Square" << std::endl;
    dax::cuda::cont::internal::SquareFunction<<<params.numPointBlocks(),
        params.numPointThreads()>>>(size,
                                    grid,
                                    inField,
                                    outField);

    //associate the resulting device array with this host array
    //the array will be pulled of the device when the user requests
    //it on the client. Otherwise it will stay on the device
    out.setExecutionArray(outd);
  }
};


class Sine
{
public:
  template<typename G, typename T, typename U>
  void operator()(G &g,
                  dax::cont::internal::ArrayContainer<T>& in,
                  U& out)
  {
    typedef typename U::ValueType OutType;

    //get the size of the field we are iterating over
    //this is derived by the output array field type (points, cells)
    dax::Id size = U::fieldSize(g);

    dax::cuda::cont::internal::DeviceArrayPtr<T> ind(locateDeviceArray(in));
    dax::cuda::cont::internal::DeviceArrayPtr<OutType> outd(
          new dax::cuda::cont::internal::DeviceArray<OutType>(size));

    assert(ind->size()==outd->size());

    //determine the cuda parameters from the data structure
    dax::cuda::exec::CudaParameters params(g);

    //this type needs to be auto derived
    dax::internal::StructureUniformGrid grid = dax::cont::internal::convert(g);

    //this should be easier to do
    dax::internal::DataArray<T> inField(ind);
    dax::internal::DataArray<OutType> outField(outd);

    std::cout << "Sine" << std::endl;
    dax::cuda::cont::internal::SineFunction<<<params.numPointBlocks(),
        params.numPointThreads()>>>(size,
                                    grid,
                                    inField,
                                    outField);

    //associate the resulting device array with this host array
    //the array will be pulled of the device when the user requests
    //it on the client. Otherwise it will stay on the device
    out.setExecutionArray(outd);
  }
};

class Cosine
{
public:
  template<typename G, typename T, typename U>
  void operator()(G &g,
                  dax::cont::internal::ArrayContainer<T>& in,
                  U& out)
  {
    typedef typename U::ValueType OutType;

    //get the size of the field we are iterating over
    //this is derived by the output array field type (points, cells)
    dax::Id size = U::fieldSize(g);

    dax::cuda::cont::internal::DeviceArrayPtr<T> ind(locateDeviceArray(in));
    dax::cuda::cont::internal::DeviceArrayPtr<OutType> outd(
          new dax::cuda::cont::internal::DeviceArray<OutType>(size));

    assert(ind->size()==outd->size());

    //determine the cuda parameters from the data structure
    dax::cuda::exec::CudaParameters params(g);

    //this type needs to be auto derived
    dax::internal::StructureUniformGrid grid = dax::cont::internal::convert(g);

    //this should be easier to do
    dax::internal::DataArray<T> inField(ind);
    dax::internal::DataArray<OutType> outField(outd);

    std::cout << "Cosine" << std::endl;
    dax::cuda::cont::internal::CosineFunction<<<params.numPointBlocks(),
        params.numPointThreads()>>>(size,
                                    grid,
                                    inField,
                                    outField);

    //associate the resulting device array with this host array
    //the array will be pulled of the device when the user requests
    //it on the client. Otherwise it will stay on the device
    out.setExecutionArray(outd);
  }
};


class CellGradient
{
public:
  template<typename G, typename T, typename T2, typename U>
  void operator()(G &g,
                  dax::cont::internal::ArrayContainer<T>& in,
                  dax::cont::internal::ArrayContainer<T2>& in2,
                  U& out)
  {
    typedef typename U::ValueType OutType;

    //get the size of the field we are iterating over
    //this is derived by the output array field type (points, cells)
    dax::Id size = U::fieldSize(g);

    dax::cuda::cont::internal::DeviceArrayPtr<T> ind(locateDeviceArray(in));
    dax::cuda::cont::internal::DeviceArrayPtr<T2> ind2(locateDeviceArray(in2));
    dax::cuda::cont::internal::DeviceArrayPtr<OutType> outd(
          new dax::cuda::cont::internal::DeviceArray<OutType>(size));


    //out is on cells, while in is on points
    assert(ind->size()==ind2->size());

    //determine the cuda parameters from the data structure
    dax::cuda::exec::CudaParameters params(g);

    //this type needs to be auto derived
    dax::internal::StructureUniformGrid grid = dax::cont::internal::convert(g);

    //this should be easier to do
    dax::internal::DataArray<T> inField(ind);
    dax::internal::DataArray<T2> inField2(ind2);
    dax::internal::DataArray<OutType> outField(outd);

    std::cout << "Cell Gradient" << std::endl;
    dax::cuda::cont::internal::CellGradientFunction<<<params.numPointBlocks(),
        params.numPointThreads()>>>(size,
                                    grid,
                                    inField,
                                    inField2,
                                    outField);

    //associate the resulting device array with this host array
    //the array will be pulled of the device when the user requests
    //it on the client. Otherwise it will stay on the device
    out.setExecutionArray(outd);
  }
};

} } } }
#endif // __dax_cuda_cont_Worklets_h
