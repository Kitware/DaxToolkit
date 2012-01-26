/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_cont_ErrorControlBadValue_h
#define __dax_cont_ErrorControlBadValue_h

#include <dax/cont/ErrorControl.h>

namespace dax {
namespace cont {

/// This class is thrown when a Dax function or method encounters an invalid
/// value that inhibits progress.
///
class ErrorControlBadValue : public ErrorControl
{
public:
  ErrorControlBadValue(const std::string &message)
    : ErrorControl(message) { }
};

}
} // namespace dax::cont

#endif //__dax_cont_ErrorControlBadValue_h
