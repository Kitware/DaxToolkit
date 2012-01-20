/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_StreamCompact_h
#define __dax_cont_StreamCompact_h

#include <dax/Types.h>
#include <vector>

namespace dax {
namespace cont {

template<typename T>
DAX_CONT_EXPORT void streamCompactDebug(const std::vector<T>& input,
                                        const std::vector<dax::Id>& stencil,
                                        std::vector<T>& output)
{
  typedef typename std::vector<T>::const_iterator Iterator;
  typedef std::vector<dax::Id>::const_iterator StencilIterator;

  output.reserve(input.size());
  StencilIterator si=stencil.begin();
  for(Iterator i=input.begin();
      i!=input.end();
      ++i,++si)
    {
    if(*si)
      {
      output.push_back(*i);
      }
    }
  //reduce the allocation request
  output.reserve(output.size());

}

DAX_CONT_EXPORT void streamCompactDebug(const std::vector<dax::Id>& input,
                                        std::vector<dax::Id>& output)
{
  typedef std::vector<dax::Id>::const_iterator Iterator;
  output.reserve(input.size());
  dax::Id index = 0;
  for(Iterator i=input.begin();i!=input.end();++i,++index)
    {
    if(*i)
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


