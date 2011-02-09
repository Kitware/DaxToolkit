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
  std::string GetName() const;
  void SetName(const std::string &name);

  /// Get/Set the port type.
  int GetType() const;
  void SetType(int);

  /// Get/Set the number of components.
  int GetNumberOfComponents() const;
  void SetNumberOfComponents(int);

  /// Returns true if this fncPort can be connected as a sink to the \c
  /// sourcePort.
  bool CanSourceFrom(const fncPort* sourcePort) const;

  enum Types
    {
    invalid=0,
    point_array,
    cell_array,
    any_array,
    float_,
    };
protected:
  std::string Name;
  int Type;
  int NumberOfComponents;

private:
  fncDisableCopyMacro(fncPort);
};

/// declares fncPortPtr
fncDefinePtrMacro(fncPort)
#endif
