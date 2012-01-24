/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_internal_Testing_h
#define __dax_internal_Testing_h

#include <dax/Types.h>

#include <iostream>
#include <sstream>
#include <string>

// Try to enforce using the correct testing version. (Those that include the
// control environment have more possible exceptions.) This is not guaranteed
// to work. To make it more likely, place the Testing.h include last.
#ifdef __dax_cont_Error_h
#ifndef __dax_cont_internal_Testing_h
#error Use dax::cont::internal::Testing instead of dax::internal::Testing.
#else
#define DAX_TESTING_IN_CONT
#endif
#endif

/// \def DAX_TEST_ASSERT(condition, message)
///
/// Asserts a condition for a test to pass. A passing condition is when \a
/// condition resolves to true. If \a condition is false, then the test is
/// aborted and failure is returned.

#define DAX_TEST_ASSERT(condition, message) \
  ::dax::internal::Testing::Assert( \
      condition, __FILE__, __LINE__, message, #condition)

/// \def DAX_TEST_FAIL(message)
///
/// Causes a test to fail with the given \a message.

#define DAX_TEST_FAIL(message) \
  throw ::dax::internal::Testing::TestFailure(__FILE__, __LINE__, message)

namespace dax {
namespace internal {

struct Testing
{
public:
  class TestFailure
  {
  public:
    DAX_EXEC_CONT_EXPORT TestFailure(const std::string &file,
                                     dax::Id line,
                                     const std::string &message)
      : File(file), Line(line), Message(message) { }

    DAX_EXEC_CONT_EXPORT TestFailure(const std::string &file,
                                     dax::Id line,
                                     const std::string &message,
                                     const std::string &condition)
      : File(file), Line(line)
    {
      this->Message.append(message);
      this->Message.append(" (");
      this->Message.append(condition);
      this->Message.append(")");
    }

    DAX_EXEC_CONT_EXPORT const std::string &GetFile() const {
      return this->File;
    }
    DAX_EXEC_CONT_EXPORT dax::Id GetLine() const { return this->Line; }
    DAX_EXEC_CONT_EXPORT const std::string &GetMessage() const {
      return this->Message;
    }
  private:
    std::string File;
    dax::Id Line;
    std::string Message;
  };

  static DAX_EXEC_CONT_EXPORT void Assert(bool condition,
                                          const std::string &file,
                                          dax::Id line,
                                          const std::string &message,
                                          const std::string &conditionString)
  {
    if (condition)
      {
      // Do nothing.
      }
    else
      {
      throw TestFailure(file, line, message, conditionString);
      }
  }

#ifndef DAX_TESTING_IN_CONT
  template<class Func>
  static DAX_EXEC_CONT_EXPORT int Run(Func function)
  {
    try
      {
      function();
      }
    catch (TestFailure error)
      {
      std::cout << "***** Test failed @ "
                << error.GetFile() << ":" << error.GetLine() << std::endl
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
#endif
};

}
} // namespace dax::internal

#endif //__dax_internal_Testing_h
