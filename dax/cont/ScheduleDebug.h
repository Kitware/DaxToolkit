/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_Schedule_h
#define __dax_cont_Schedule_h

#include <dax/Types.h>

namespace dax {
namespace cont {

template<class Functor, class Parameters>
DAX_CONT_EXPORT void scheduleDebug(Functor functor,
                                   Parameters &parameters,
                                   dax::Id numInstances)
{
  for (dax::Id index = 0; index < numInstances; index++)
    {
    functor(parameters, index);
    }
}

}
} // namespace dax::cont

#endif //__dax_cont_Schedule_h
