/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_internal_Testing_h
#define __dax_cont_internal_Testing_h

#include <dax/cont/Error.h>

#include <dax/internal/Testing.h>

namespace dax {
namespace cont {
namespace internal {

struct Testing
{
public:
  template<class Func>
  static DAX_EXEC_CONT_EXPORT int Run(Func function)
  {
    try
      {
      function();
      }
    catch (dax::internal::Testing::TestFailure error)
      {
      std::cout << "***** Test failed @ "
                << error.GetFile() << ":" << error.GetLine() << std::endl
                << error.GetMessage() << std::endl;
      return 1;
      }
    catch (dax::cont::Error error)
      {
      std::cout << "***** Uncaught Dax exception thrown." << std::endl
                << error.GetMessage() << std::endl;
      return 1;
      }
    catch (...)
      {
      std::cout << "***** Unidentified exception thrown." << std::endl;
      return 1;
      }
    return 0;
  }
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_Testing_h
