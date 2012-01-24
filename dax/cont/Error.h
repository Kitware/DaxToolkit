/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_cont_Error_h
#define __dax_cont_Error_h

// Note that this class and (most likely) all of its subclasses are not
// templated.  If there is any reason to create a Dax control library,
// this class and its subclasses should probably go there.

#include <string>

namespace dax {
namespace cont {

/// The superclass of all exceptions thrown by any Dax function or method.
///
class Error
{
public:
  const std::string &GetMessage() const { return this->Message; }

protected:
  Error() { }
  Error(const std::string message) : Message(message) { }

  void SetMessage(const std::string &message) {
    this->Message = message;
  }

private:
  std::string Message;
};

}
} // namespace dax::cont

#endif //__dax_cont_Error_h
