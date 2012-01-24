/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_cont_ErrorControlAssert_h
#define __dax_cont_ErrorControlAssert_h

#include <dax/Types.h>
#include <dax/cont/ErrorControl.h>

#include <stdio.h>

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
    char lineString[32];
    sprintf(lineString, "%d", this->Line);

    std::string message;
    message.append(this->File);
    message.append(":");
    message.append(lineString);
    message.append(": Assert Failed (");
    message.append(this->Condition);
    message.append(")");
    this->SetMessage(message);
  }

private:
  std::string File;
  dax::Id Line;
  std::string Condition;
};

}
}

#endif //__dax_cont_ErrorControlAssert_h
