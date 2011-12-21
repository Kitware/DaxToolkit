#ifndef __dax_cuda_cont_Worklets_h
#define __dax_cuda_cont_Worklets_h


#include <cuda.h>

#include <dax/Types.h>

#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>

#include <dax/cont/StructuredGrid.h>
#include <dax/cont/internal/ConvertTypes.h>

#include <dax/cuda/cont/internal/DeviceArray.h>
#include <dax/cuda/cont/internal/WorkletFunctions.h>
#include <dax/cuda/exec/CudaParameters.h>


namespace dax { namespace cuda {  namespace cont { namespace worklet {

class Elevation
{
public:
  template<typename G, typename T, typename U>
  void operator()(G &g,
                  const dax::cont::Array<T>& in,
                  U& out)
  {
    typedef typename U::DataType OutType;

    //convert from host to device arrayss
    dax::cuda::cont::internal::DeviceArray<T> ind(in);
    dax::cuda::cont::internal::DeviceArray<OutType> outd(out.array().size());

    //determine the cuda parameters from the data structure
    dax::cuda::exec::CudaParameters params(g);

    //get the size of the field we are iterating over
    //this is derived by the output array field type (points, cells)
    dax::Id size = U::fieldSize(g);

    //this type needs to be auto derived
    dax::internal::StructureUniformGrid grid = dax::cont::internal::convert(g);

    //you can't use implicit constructors when calling a global
    //function
    dax::internal::DataArray<T> inField(ind);
    dax::internal::DataArray<OutType> outField(outd);
    dax::cuda::cont::internal::ElevationFunction<<<params.numPointBlocks(),
        params.numPointThreads()>>>(size,
                                    grid,
                                    inField,
                                    outField);

    //move the results back from the device to the host
    outd.toHost(&out.array());
  }
};

class Square
{
public:
  template<typename G, typename T, typename U>
  void operator()(G &g,
                  const dax::cont::Array<T>& in,
                  U& out)
  {
    typedef typename U::DataType OutType;

    //convert from host to device arrayss
    dax::cuda::cont::internal::DeviceArray<T> ind(in);
    dax::cuda::cont::internal::DeviceArray<OutType> outd(out.array());

    //determine the cuda parameters from the data structure
    dax::cuda::exec::CudaParameters params(g);

    //this type needs to be auto derived
    dax::internal::StructureUniformGrid grid = dax::cont::internal::convert(g);

    //this should be easier to do
    dax::internal::DataArray<T> inField(ind);
    dax::internal::DataArray<OutType> outField(outd);

    //get the size of the field we are iterating over
    //this is derived by the output array field type (points, cells)
    dax::Id size = U::fieldSize(g);

    //in and out are automatically converted to the correct type
    //by explicit constructors on FieldPoint and FieldCoordinates
    dax::cuda::cont::internal::SquareFunction<<<params.numPointBlocks(),
        params.numPointThreads()>>>(size,
                                    grid,
                                    inField,
                                    outField);

   //move the results back from the device to the host
    out.array() = outd;
  }
};


class Sine
{
public:
  template<typename G, typename T, typename U>
  void operator()(G &g,
                  const dax::cont::Array<T>& in,
                  U& out)
  {
    typedef typename U::DataType OutType;

    //convert from host to device arrayss
    dax::cuda::cont::internal::DeviceArray<T> ind(in);
    dax::cuda::cont::internal::DeviceArray<OutType> outd(out.array());

    //determine the cuda parameters from the data structure
    dax::cuda::exec::CudaParameters params(g);

    //this type needs to be auto derived
    dax::internal::StructureUniformGrid grid = dax::cont::internal::convert(g);

    //this should be easier to do
    dax::internal::DataArray<T> inField(ind);
    dax::internal::DataArray<OutType> outField(outd);

    //get the size of the field we are iterating over
    //this is derived by the output array field type (points, cells)
    dax::Id size = U::fieldSize(g);

    //in and out are automatically converted to the correct type
    //by explicit constructors on FieldPoint and FieldCoordinates
    dax::cuda::cont::internal::SineFunction<<<params.numPointBlocks(),
        params.numPointThreads()>>>(size,
                                    grid,
                                    inField,
                                    outField);

   //move the results back from the device to the host
    out.array() = outd;
  }
};

class Cosine
{
public:
  template<typename G, typename T, typename U>
  void operator()(G &g,
                  const dax::cont::Array<T>& in,
                  U& out)
  {
    typedef typename U::DataType OutType;

    //convert from host to device arrayss
    dax::cuda::cont::internal::DeviceArray<T> ind(in);
    dax::cuda::cont::internal::DeviceArray<OutType> outd(out.array());

    //determine the cuda parameters from the data structure
    dax::cuda::exec::CudaParameters params(g);

    //this type needs to be auto derived
    dax::internal::StructureUniformGrid grid = dax::cont::internal::convert(g);

    //this should be easier to do
    dax::internal::DataArray<T> inField(ind);
    dax::internal::DataArray<OutType> outField(outd);

    //get the size of the field we are iterating over
    //this is derived by the output array field type (points, cells)
    dax::Id size = U::fieldSize(g);

    //in and out are automatically converted to the correct type
    //by explicit constructors on FieldPoint and FieldCoordinates
    dax::cuda::cont::internal::CosineFunction<<<params.numPointBlocks(),
        params.numPointThreads()>>>(size,
                                    grid,
                                    inField,
                                    outField);

   //move the results back from the device to the host
    out.array() = outd;
  }
};


class CellGradient
{
public:
  template<typename G, typename T, typename T2, typename U>
  void operator()(G &g,
                  const dax::cont::Array<T>& in,
                  const dax::cont::Array<T2>& in2,
                  U& out)
  {
    typedef typename U::DataType OutType;

    //convert from host to device arrayss
    dax::cuda::cont::internal::DeviceArray<T> ind(in);
    dax::cuda::cont::internal::DeviceArray<T2> ind2(in2);
    dax::cuda::cont::internal::DeviceArray<OutType> outd(out.array());

    //determine the cuda parameters from the data structure
    dax::cuda::exec::CudaParameters params(g);

    //this type needs to be auto derived
    dax::internal::StructureUniformGrid grid = dax::cont::internal::convert(g);

    //this should be easier to do
    dax::internal::DataArray<T> inField(ind);
    dax::internal::DataArray<T2> inField2(ind2);
    dax::internal::DataArray<OutType> outField(outd);

    //get the size of the field we are iterating over
    //this is derived by the output array field type (points, cells)
    dax::Id size = U::fieldSize(g);

    //in and out are automatically converted to the correct type
    //by explicit constructors on FieldPoint and FieldCoordinates
    dax::cuda::cont::internal::CellGradientFunction<<<params.numPointBlocks(),
        params.numPointThreads()>>>(size,
                                    grid,
                                    inField,
                                    inField2,
                                    outField);

    //move the results back from the device to the host
    out.array() = outd;
  }
};

} } } }
#endif // __dax_cuda_cont_Worklets_h
