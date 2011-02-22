/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxPort_h
#define __daxPort_h

#include "daxObject.h"

/// daxPort represents an input or an output port for a module. It contains all
/// necessary information to determine the kind of input expected on the port,
/// the name of the port etc.
class daxPort : public daxObject
{
public:
  daxPort();
  virtual ~daxPort();
  daxTypeMacro(daxPort, daxObject);

  /// Get/Set the port name.
  std::string GetName() const;
  void SetName(const std::string &name);

  /// Returns true if this daxPort can be connected as a sink to the \c
  /// sourcePort.
  bool CanSourceFrom(const daxPort* sourcePort) const;

protected:
  std::string Name;

private:
  daxDisableCopyMacro(daxPort);
};

/// declares daxPortPtr
daxDefinePtrMacro(daxPort)
#endif
