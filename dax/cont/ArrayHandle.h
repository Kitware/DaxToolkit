/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_ArrayHandle_h
#define __dax_cont_ArrayHandle_h

#include <dax/Types.h>

#include <dax/cont/internal/IteratorContainerPolymorphic.h>

namespace dax {
namespace cont {

template<typename T>
class ArrayHandle
{
public:
  typedef T ValueType;

  dax::Id GetNumberOfEntries() const { return this->NumberOfEntries; }
private:
  dax::Id NumberOfEntries;
};

}
}

#endif //__dax_cont_ArrayHandle_h
