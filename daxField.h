/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxField_h
#define __daxField_h

#include "daxObject.h"

class daxArray;
daxDefinePtrMacro(daxArray);

/// daxField represents a fundamental object in the host Data Model. daxField
/// consists of one or more named daxArrays. Using ref and dep attributes, one
/// can define connections, point attributes, cell attributes etc. And using
/// different types of daxArray subclasses, one can define regular grids as well
/// as unstructured grids.
class daxField : public daxObject
{
public:
  daxField();
  virtual ~daxField();
  daxTypeMacro(daxField, daxObject);

  /// Get a named component.
  daxArrayPtr GetComponent(const char* name) const;

  /// Set a named component.
  void SetComponent(const char* name, daxArrayPtr array);

  /// Returns true is the named component exists.
  bool HasComponent(const char* name) const;

private:
  daxDisableCopyMacro(daxField);
  class daxInternals;
  daxInternals* Internals;
};

/// defines weak_ptr and shared_ptr types.
daxDefinePtrMacro(daxField);

#endif
