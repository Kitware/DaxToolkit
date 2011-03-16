/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxExecutive2_h
#define __daxExecutive2_h

#include "daxObject.h"

class daxModule;
class daxPort;
class daxDataObject;
daxDefinePtrMacro(daxModule);
daxDefinePtrMacro(daxPort);

/// daxExecutive2 is the executive. One typically creates a single executive and
/// then registers module connections with that execute. To trigger an
/// execution, one finally calls daxExecutive2::Execute().
///
/// In the first pass, we are only going to work with uniform-rectilinear grids.
/// So we provide explicit API to set the grid to work on. In future that will
/// change. We need to spend more time on the data-model, but we'll do that
/// later.
class daxExecutive2 : public daxObject
{
public:
  daxExecutive2();
  virtual ~daxExecutive2();
  daxTypeMacro(daxExecutive2, daxObject);

  /// Register a connection. Returns true if the connection was setup correctly.
  bool Connect(
    const daxModulePtr sourceModule, const daxPortPtr sourcePort,
    const daxModulePtr sinkModule, const daxPortPtr sinkPort);
  bool Connect(
    const daxModulePtr sourceModule, const std::string& sourcename,
    const daxModulePtr sinkModule, const std::string& sinkname);

  /// Resets the executive. This is the only way to break connections until we
  /// start supporting de-connecting.
  void Reset();

  void PrintKernel();
  std::string GetKernel();
protected:
  /// Executes every subtree in the graph separately. We currently invoke every
  /// subtree as a separate kernel. We can merge the kernels or something of
  /// that sort in future.
  template <class Graph>
    bool ExecuteOnce(
      typename Graph::vertex_descriptor head, const Graph& graph) const;

private:
  daxDisableCopyMacro(daxExecutive2);

  class daxInternals;
  daxInternals* Internals;
};

daxDefinePtrMacro(daxExecutive2);
#endif
