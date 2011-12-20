/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_Filter_h
#define __dax_cuda_cont_Filter_h

#include <dax/Types.h>
#include <dax/cont/internal/FilterBase.h>
#include <dax/cont/Array.h>

#include "Model.h"
#include "Modules.h"


//Add to the standard control enviornment the Filters for cuda
//since we are including cuda headers
namespace dax { namespace cont {

template<typename Function>
class Filter <dax::cuda::cont::MapFieldModule<Function> > :
    public dax::cont::internal::FilterBase< Filter <dax::cuda::cont::MapFieldModule<Function> > >
{
public:
  typedef typename dax::cuda::cont::MapFieldModule<Function>::OutputType OutputType;


  //Default constructor is to operate on points
  template<typename T>
  Filter(dax::cuda::cont::Model<T> &model):
    Module(model,dax::cont::PointField())
  {
    this->AlreadyComputed = false;
  }


  template<typename T, typename FieldType>
  Filter(dax::cuda::cont::Model<T> &model, const FieldType&):
    Module(model,FieldType())
  {
    this->AlreadyComputed = false;
  }

  void pullResults(dax::cont::Array<OutputType> &data)
  {
    data = Module.output().get();
  }

  void compute()
  {
    Module.compute();
  }

  dax::cuda::cont::MapFieldModule<Function> Module;
};

template<typename Function>
class Filter <dax::cuda::cont::MapCellModule<Function> > :
    public dax::cont::internal::FilterBase< Filter <dax::cuda::cont::MapCellModule<Function> > >
{
public:
  typedef typename dax::cuda::cont::MapCellModule<Function>::OutputType OutputType;

  template<typename T, typename F>
  Filter(dax::cuda::cont::Model<T> &model,
         dax::cont::Filter<dax::cuda::cont::MapFieldModule<F> > &filter):
    Module(model,filter.Module)
  {
    this->addDependency(filter);
    this->AlreadyComputed = false;
  }

  void pullResults(dax::cont::Array<OutputType> &data)
  {
    data = Module.output().get();
  }

  void compute()
  {
    Module.compute();
  }

  dax::cuda::cont::MapCellModule<Function> Module;
};

} }

#endif
