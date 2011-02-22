/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxCellAverageModule_h
#define __daxCellAverageModule_h

#include "daxModule.h"

/// Temporary module until we start supporting auto-generation of modules from
/// the functor.
class daxCellAverageModule : public daxModule
{
public:
  daxCellAverageModule();

  /// Returns the name for this module.
  virtual const std::string& GetModuleName() const
    { return this->Name; }

  /// Returns the functor code.
  virtual const std::string& GetFunctorCode() const
    { return this->Code; }

private:
  daxDefinePtrMacro(daxCellAverageModule);
  std::string Code;
  std::string Name;
};

daxDefinePtrMacro(daxCellAverageModule);

#endif
