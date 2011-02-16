/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxExecutive_h
#define __daxExecutive_h

#include "daxObject.h"

class daxImageData;
class daxModule;
class daxPort;
daxDefinePtrMacro(daxImageData);
daxDefinePtrMacro(daxModule);
daxDefinePtrMacro(daxPort);

/// daxExecutive is the executive. One typically creates a single executive and
/// then registers module connections with that execute. To trigger an
/// execution, one finally calls daxExecutive::Execute().
///
/// In the first pass, we are only going to work with uniform-rectilinear grids.
/// So we provide explicit API to set the grid to work on. In future that will
/// change. We need to spend more time on the data-model, but we'll do that
/// later.
class daxExecutive : public daxObject
{
public:
  daxExecutive();
  virtual ~daxExecutive();
  daxTypeMacro(daxExecutive, daxObject);

  /// Executes the pipelines. Return false if there's some error.
  /// Initially, we'll assume 1 input and 1 output with only 1 pipeline defined.
  bool Execute(const daxImageData* input, daxImageData* output) const;

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

protected:
  /// Executes every subtree in the graph separately. We currently invoke every
  /// subtree as a separate kernel. We can merge the kernels or something of
  /// that sort in future.
  template <class Graph, class InputDataType, class OutputDataType>
    bool ExecuteOnce(
      typename Graph::vertex_descriptor head, const Graph& graph,
      const InputDataType* input, OutputDataType* output) const;

private:
  daxDisableCopyMacro(daxExecutive);

  class daxInternals;
  daxInternals* Internals;
};

daxDefinePtrMacro(daxExecutive);

#include <string>
#include <map>

namespace dax
{
  std::string daxSubstituteKeywords(
    const char* source, const std::map<std::string, std::string>& keywords);
}

#endif
