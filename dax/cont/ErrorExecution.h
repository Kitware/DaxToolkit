/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_cont_ErrorExecution_h
#define __dax_cont_ErrorExecution_h

#include <dax/cont/Error.h>

namespace dax {
namespace cont {

/// This class is thrown in the control environment whenever an error occurs in
/// the execution environment.
///
class ErrorExecution : public dax::cont::Error
{
public:
  ErrorExecution(const std::string message)
    : Error(message) { }
};

}
}  // namespace dax::cont

#endif //__dax_cont_ErrorExecution_h
