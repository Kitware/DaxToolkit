/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_cont_ErrorControl_h
#define __dax_cont_ErrorControl_h

#include <dax/cont/Error.h>

namespace dax {
namespace cont {

/// The superclass of all exceptions thrown from within the Dax control
/// environment.
///
class ErrorControl : public dax::cont::Error
{
public:
  ErrorControl() { }
  ErrorControl(const std::string message) : Error(message) { }
};

}
} // namespace dax::cont

#endif //__dax_cont_ErrorControl_h
