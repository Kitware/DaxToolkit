#ifndef __dax_cuda_cont_MapFieldModule_h
#define __dax_cuda_cont_MapFieldModule_h

#include <dax/cont/FilterFieldTypes.h>
#include <dax/cont/DataSet.h>

#include <dax/cuda/cont/Model.h>
#include <dax/cuda/cont/internal/DeviceArray.h>
#include <dax/cuda/exec/CudaParameters.h>
#include <vector>

namespace dax { namespace cuda { namespace cont {

template< typename Worklet >
class MapFieldModule
{
public:

  typedef typename dax::cont::DataSet ModelType;
  typedef typename Worklet::OutputType OutputType;
  typedef dax::cuda::cont::internal::DeviceArray
                                      <OutputType> OutputDataArray;
  typedef dax::cuda::cont::internal::DeviceArrayPtr
                                      <OutputType> OutputDataArrayPtr;

  template <typename MT, typename Field_Type>
  MapFieldModule(dax::cuda::cont::Model<MT> &m, const Field_Type&):
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
    this->Output->resize(dax::cont::FieldSize(this->FieldType,this->Model));

    const dax::cuda::exec::CudaParameters params(*this->Model);

    //uses implicit constructor conversion from
    //DeviceArray to DataArray
    Worklet().run(params,*this->Model,this->Output);
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
