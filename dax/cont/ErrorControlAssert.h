/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_cont_ErrorControlAssert_h
#define __dax_cont_ErrorControlAssert_h

#include <dax/Types.h>
#include <dax/cont/ErrorControl.h>

#include <sstream>

namespace dax {
namespace cont {

/// This error is thrown whenever DAX_ASSERT_CONT fails.
///
class ErrorControlAssert : public dax::cont::ErrorControl
{
public:
  ErrorControlAssert(const std::string &file,
                     dax::Id line,
                     const std::string &condition)
    : ErrorControl(), File(file), Line(line), Condition(condition)
  {
    std::stringstream message;
    message << this->File << ":" << this->Line
            << ": Assert Failed (" << this->Condition << ")";
    this->SetMessage(message.str());
  }

private:
  std::string File;
  dax::Id Line;
  std::string Condition;
};

}
}

#endif //__dax_cont_ErrorControlAssert_h
