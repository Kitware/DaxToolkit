/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __fncModule_h
#define __fncModule_h

#include "fncObject.h"

class fncPort;
fncDefinePtrMacro(fncPort)

/// fncModule defines a module. Modules are required to define functors and
/// connect them in pipelines. Typically modules are auto-generated using
/// functor source code prefixed with a module description header (MDF).
/// Alternatively, developers can explicitly subclass this class to define
/// modules of their own.
class fncModule : public fncObject
{
public:
  fncModule();
  virtual ~fncModule();
  fncTypeMacro(fncModule, fncObject);

  /// Returns the name for this module.
  virtual const std::string& GetModuleName() = 0;

  /// Returns the functor code.
  virtual const std::string& GetFunctorCode() = 0;

  /// Returns the number of inputs needed by this module.
  virtual size_t GetNumberOfInputs();

  /// Returns the name of the input port at the given index.
  virtual std::string GetInputPortName(size_t index);

  /// Returns the input information information object for the particular input.
  /// Returns NULL if the index is out-of-range.
  virtual fncPortPtr GetInputPort(size_t index);

  /// Returns the input port information given the name of the port.
  virtual fncPortPtr GetInputPort(const std::string& portname);

  /// Returns the number of outputs needed by this module.
  virtual size_t GetNumberOfOutputs();

  /// Returns the name of the output port at the given index.
  virtual std::string GetOutputPortName(size_t index);

  /// Returns the output information information object for the particular output.
  /// Returns NULL if the index is out-of-range.
  virtual fncPortPtr GetOutputPort(size_t index);

  /// Returns the output port information given the name of the port.
  virtual fncPortPtr GetOutputPort(const std::string& portname);

protected:

  /// Set the number of inputs.
  void SetNumberOfInputs(size_t count);

  /// Add an input port information.
  void SetInputPort(size_t index, fncPortPtr port);

  /// Set the number of outputs.
  void SetNumberOfOutputs(size_t count);

  /// Add an output port information.
  void SetOutputPort(size_t index, fncPortPtr port);

private:
  fncDisableCopyMacro(fncModule);

  class fncInternals;
  fncInternals* Internals;
};

/// declares fncModulePtr
fncDefinePtrMacro(fncModule)

#endif
