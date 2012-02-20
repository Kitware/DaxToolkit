/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_StreamCompact_h
#define __dax_cont_StreamCompact_h

#include <dax/Types.h>
#include <dax/Functional.h>

#include <dax/cont/internal/ArrayContainerExecutionCPU.h>
#include <dax/internal/GridTopologys.h>

#include <algorithm>

namespace dax {
namespace cont {

template<typename T, typename U>
DAX_EXEC_CONT_EXPORT void streamCompactDebug(
    const dax::cont::internal::ArrayContainerExecutionCPU<T>& input,
    const dax::cont::internal::ArrayContainerExecutionCPU<U>& stencil,
    dax::cont::internal::ArrayContainerExecutionCPU<T>& output)
{
  typedef typename dax::cont::internal::ArrayContainerExecutionCPU<T>::const_iterator CIter;
  typedef typename dax::cont::internal::ArrayContainerExecutionCPU<T>::iterator Iter;

  //first count the number of values to go into the output array.
  dax::Id size = std::count_if(stencil.begin(),
                               stencil.end(),
                               dax::not_identity<U>());
  output.Allocate(size);
  Iter out = output.begin();
  Iter outEnd = output.end();
  CIter in=input.begin();
  CIter sten = stencil.begin();
  for(;out!=outEnd;++in,++sten)
    {
    //check the output so we loop over the smaller sized array
    if(dax::not_identity<U>()(*sten))
      {
      //only remove cell that match the identity ( aka default constructor ) of T
      *out = *in;
      ++out;
      }
    }
}

template<typename T,typename U>
DAX_EXEC_CONT_EXPORT void streamCompactDebug(
    const dax::cont::internal::ArrayContainerExecutionCPU<T>& input,
    dax::cont::internal::ArrayContainerExecutionCPU<U>& output)
{
  typedef typename dax::cont::internal::ArrayContainerExecutionCPU<T>::const_iterator CIter;
  typedef typename dax::cont::internal::ArrayContainerExecutionCPU<U>::iterator Iter;

  //first count the number of values to go into the output array.
  dax::Id size = std::count_if(input.begin(),
                               input.end(),
                               dax::not_identity<T>());

  output.Allocate(size);
  Iter out = output.begin();
  Iter outEnd = output.end();
  CIter in=input.begin();
  U index = U();
  for(;out!=outEnd;++in,++index)
    {
    //check the output so we loop over the smaller sized array
    if(dax::not_identity<T>()(*in))
      {
      //only remove cell that match the identity ( aka default constructor ) of T
      *out = index;
      ++out;
      }
    }
}


template<typename GridType, typename T, typename U>
DAX_EXEC_CONT_EXPORT void streamCompactTopologyDebug(
    const GridType& inputGrid,
    dax::internal::TopologyUnstructured<T>& output,
    const dax::cont::internal::ArrayContainerExecutionCPU<U>& cellIds,
    const dax::cont::internal::ArrayContainerExecutionCPU<U>& pointIds)
{

  //cellIds is a collection of all the ids we need in the output grid,
  //and point ids are the collection of points we need.
  //The major isssue is that we need to construct a topology array
  //that maps the old pointIds to the newPointIds for the cells.

}



}
} // namespace dax::cont

#endif //__dax_cont_StreamCompact_h


