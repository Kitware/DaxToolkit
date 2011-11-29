#ifndef __dax_cuda_cont_MapCellModule_h
#define __dax_cuda_cont_MapCellModule_h

#include <dax/cont/FilterFieldTypes.h>
#include <dax/cuda/cont/Model.h>
#include <dax/cuda/cont/internal/DeviceArray.h>
#include <dax/cuda/exec/CudaParameters.h>
#include <vector>

namespace dax { namespace cuda { namespace cont {

template< typename Worklet>
class MapCellModule
{
public:
  typedef typename Worklet::ModelType ModelType;
  typedef typename Worklet::InputType InputType;
  typedef typename Worklet::OutputType OutputType;

  typedef dax::cuda::cont::internal::DeviceArray
                                      <InputType> InputDataArray;
  typedef dax::cuda::cont::internal::DeviceArrayPtr
                                      <InputType> InputDataArrayPtr;


  typedef dax::cuda::cont::internal::DeviceArray
                                      <OutputType> OutputDataArray;
  typedef dax::cuda::cont::internal::DeviceArrayPtr
                                      <OutputType> OutputDataArrayPtr;

  template <typename Module >
  MapCellModule(dax::cuda::cont::Model<ModelType> &m, const Module &module):
    Input(module.output()),
    Output( new OutputDataArray() ),
    FieldType( new dax::cont::CellField()),
    DeviceType(new dax::cont::CudaDevice()),
    Model(&(m.Data))
    {
    this->Model = &(m.Data);
    }

  ~MapCellModule()
  {
    delete FieldType;
    delete DeviceType;
  }

  void compute()
  {
    this->Output->resize(dax::cont::FieldSize(this->FieldType,this->Model));

    const dax::cuda::exec::CudaParameters params(*this->Model);

    //uses implicit constructor conversion from
    //DeviceArray to DataArray
    Worklet().run(params,*this->Model,
                  this->Input,
                  this->Output);
  }


  OutputDataArrayPtr output() { return Output; }

protected:
  InputDataArrayPtr Input;
  OutputDataArrayPtr Output;
  ModelType *Model;

  dax::cont::FieldType *FieldType;
  const dax::cont::DeviceType *DeviceType;
};

} } }
#endif
