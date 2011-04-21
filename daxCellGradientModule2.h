/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxCellGradientModule2_h
#define __daxCellGradientModule2_h

#include "daxModule.h"

/// Temporary module until we start supporting auto-generation of modules from
/// the functor.
class daxCellGradientModule2 : public daxModule
{
public:
  daxCellGradientModule2();

  /// Returns the name for this module.
  virtual const std::string& GetModuleName() const
    { return this->Name; }

  /// Returns the functor code.
  virtual const std::string& GetFunctorCode() const
    { return this->Code; }

private:
  daxDefinePtrMacro(daxCellGradientModule2);
  std::string Code;
  std::string Name;
};

daxDefinePtrMacro(daxCellGradientModule2);

#endif
