/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __fncExecutive_h
#define __fncExecutive_h

#include "fncObject.h"

class fncModule;
class fncPort;
fncDefinePtrMacro(fncModule);
fncDefinePtrMacro(fncPort);

/// fncExecutive is the executive. One typically creates a single executive and
/// then registers module connections with that execute. To trigger an
/// execution, one finally calls fncExecutive::Execute().
class fncExecutive : public fncObject
{
public:
  fncExecutive();
  virtual ~fncExecutive();
  fncTypeMacro(fncExecutive, fncObject);

  /// Executes the pipelines. Return false if there's some error.
  bool Execute();

  /// Register a connection.
  void Connect(
    fncModulePtr sourceModule, fncPortPtr sourcePort,
    fncModulePtr sinkModule, fncPortPtr sinkPort);
  void Connect(
    fncModulePtr sourceModule, const std::string& sourcename,
    fncModulePtr sinkModule, const std::string& sinkname);

private:
  fncDisableCopyMacro(fncExecutive);
};

fncDefinePtrMacro(fncExecutive);

#endif
