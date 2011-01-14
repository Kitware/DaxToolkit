/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __fncPort_h
#define __fncPort_h

#include "fncObject.h"

/// fncPort represents an input or an output port for a module. It contains all
/// necessary information to determine the kind of input expected on the port,
/// the name of the port etc.
class fncPort : public fncObject
{
public:
  fncPort();
  virtual ~fncPort();
  fncTypeMacro(fncPort, fncObject);

  /// Get/Set the port name.
  std::string GetName();
  void SetName(const std::string &name);

  /// Get/Set the port type.
  int GetType();
  void SetType(int);

protected:
  std::string Name;
  int Type;

private:
  fncDisableCopyMacro(fncPort);
};

/// declares fncPortPtr
fncDefinePtrMacro(fncPort)
#endif
