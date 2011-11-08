/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_Executive_h
#define __dax_cuda_cont_Executive_h

#include <dax/Types.h>
#include <vector>

#include "Filter.h"

namespace dax { namespace cuda { namespace cont {

template<typename F>
void Pull(F &filter,
          std::vector<typename F::OutputType> &data)
{
  filter.execute();
  filter.pullResults(data);
  std::cout << "Properly copied data back to host" << std::endl;
}

} } }

#endif
