#ifndef __dax_cuda_cont_MapFieldModule_h
#define __dax_cuda_cont_MapFieldModule_h

#include <dax/cont/FilterFieldTypes.h>
#include <dax/cuda/cont/Model.h>
#include <dax/cuda/cont/internal/ManagedDeviceDataArray.h>
#include <dax/cuda/exec/CudaParameters.h>
#include <vector>

namespace dax { namespace cuda { namespace cont {

template< typename Worklet >
class MapFieldModule
{
public:

  typedef typename Worklet::ModelType ModelType;
  typedef typename Worklet::OutputType OutputType;
  typedef dax::cuda::cont::internal::ManagedDeviceDataArray
                                      <OutputType> OutputDataArray;
  typedef dax::cuda::cont::internal::ManagedDeviceDataArrayPtr
                                      <OutputType> OutputDataArrayPtr;

  template <typename Field_Type>
  MapFieldModule(dax::cuda::cont::Model<ModelType> &m, const Field_Type&):
    Output( new OutputDataArray() ),
    FieldType( new Field_Type()),
    DeviceType(new dax::cont::CudaDevice()),
    Model(&(m.Data))
    {
    }

  ~MapFieldModule()
  {
    delete FieldType;
    delete DeviceType;
    Model=NULL;
  }

  void compute()
  {
    //uggh finding the size is nasty since we can change it at run time
    this->Output->Allocate(dax::cont::FieldSize(this->FieldType,this->Model));

    const dax::cuda::exec::CudaParameters params(*this->Model);
    Worklet().run(params,*this->Model,this->Output->GetArray());
  }

  void pullResults(std::vector<OutputType> &results)
    {

    results.resize(dax::cont::FieldSize(this->FieldType,this->Model));

    this->Output->CopyToHost(results);
    }

  OutputDataArrayPtr output() const { return Output; }

protected:
  OutputDataArrayPtr Output;

  ModelType *Model;

  dax::cont::FieldType *FieldType;
  const dax::cont::DeviceType *DeviceType;
};

} } }
#endif
