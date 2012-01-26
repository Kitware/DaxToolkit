/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_StreamCompact_h
#define __dax_cont_StreamCompact_h

#include <dax/Types.h>
#include <dax/Functional.h>

#include <vector>

namespace dax {
namespace cont {

template<typename T, typename U>
DAX_EXEC_CONT_EXPORT void streamCompactDebug(const std::vector<T>& input,
                        const std::vector<U>& stencil,
                        std::vector<T>& output)
{
  typedef typename std::vector<T>::const_iterator Iterator;
  typedef typename std::vector<U>::const_iterator StencilIterator;

  output.reserve(input.size());
  StencilIterator si=stencil.begin();
  for(Iterator i=input.begin();
      i!=input.end();
      ++i,++si)
    {
    //only remove cell that match the identity ( aka default constructor ) of U
    if(dax::not_identity<U>()(*si))
      {
      output.push_back(*i);
      }
    }
  //reduce the allocation request
  output.reserve(output.size());
}

template<typename T>
DAX_EXEC_CONT_EXPORT void streamCompactDebug(const std::vector<T>& input,
                        std::vector<T>& output)
{
  typedef typename std::vector<T>::const_iterator Iterator;

  output.reserve(input.size());
  dax::Id index = 0;
  for(Iterator i=input.begin();i!=input.end();++i,++index)
    {
    //only remove cell that match the identity ( aka default constructor ) of T
    if(dax::not_identity<T>()(*i))
      {
      output.push_back(index);
      }
    }
  //reduce the allocation request
  output.reserve(output.size());
}


}
} // namespace dax::cont

#endif //__dax_cont_StreamCompact_h


