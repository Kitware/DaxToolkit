/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxExecutive2.h"

// Generated headers.
#include "Kernel.tmpl.h"
#include "dAPI.cl.h"

// dax headers
#include "daxModule.h"
#include "daxOptions.h"
#include "daxPort.h"

// OpenCL headers
#ifdef FNC_ENABLE_OPENCL
//# define __CL_ENABLE_EXCEPTIONS
# include <CL/cl.hpp>
# include "opecl_util.h"
#endif

// Google CTemplate Includes
#include <ctemplate/template.h>  

// external headers
#include <algorithm>
#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>
#include <boost/config.hpp>
#include <boost/format.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <utility>                   // for std::pair

class daxExecutive2::daxInternals
{
public:
  /// Meta-data stored with each vertex in the graph.
  struct VertexProperty
    {
    daxModulePtr Module;
    };

  /// Meta-data stored with each edge in the graph.
  struct EdgeProperty
    {
    daxPortPtr ProducerPort;
    daxPortPtr ConsumerPort;
    };

  typedef boost::adjacency_list<boost::vecS,
          /* don't use setS for OutEdgeList since we can have parallel edges
           * when multiple ports are presents */
          boost::vecS,
          boost::bidirectionalS,
          /* Vertex Property*/
          VertexProperty,
          /* Edge Property*/
          EdgeProperty > Graph;

  typedef boost::graph_traits<Graph>::vertex_iterator VertexIterator;
  typedef boost::graph_traits<Graph>::edge_iterator EdgeIterator;
  Graph Connectivity;
};

namespace dax
{
  /// Fills up \c heads with the vertex ids for vertices with in-degree==0.
  template <class GraphType>
  inline void get_heads(
    std::vector<typename GraphType::vertex_descriptor> &heads,
    const GraphType& graph)
    {
    typedef typename boost::graph_traits<GraphType>::vertex_iterator VertexIterator;
    std::pair<VertexIterator, VertexIterator> viter = boost::vertices(graph);
    for (; viter.first != viter.second; ++viter.first)
      {
      if (boost::in_degree(*viter.first, graph) == 0)
        {
        heads.push_back(*viter.first);
        }
      }
    }
}

//-----------------------------------------------------------------------------
daxExecutive2::daxExecutive2()
{
  this->Internals = new daxInternals();
}

//-----------------------------------------------------------------------------
daxExecutive2::~daxExecutive2()
{
  delete this->Internals;
  this->Internals = NULL;
}

//-----------------------------------------------------------------------------
bool daxExecutive2::Connect(
  const daxModulePtr sourceModule, const std::string& sourcename,
  const daxModulePtr sinkModule, const std::string& sinkname)
{
  return this->Connect(sourceModule, sourceModule->GetOutputPort(sourcename),
    sinkModule, sinkModule->GetInputPort(sinkname));
}

//-----------------------------------------------------------------------------
bool daxExecutive2::Connect(
  const daxModulePtr sourceModule, const daxPortPtr sourcePort,
  const daxModulePtr sinkModule, const daxPortPtr sinkPort)
{
  assert(sourceModule && sourcePort && sinkModule && sinkPort);
  assert(sourceModule != sinkModule);

  // * Validate that their types match up.
  if (sinkPort->CanSourceFrom(sourcePort.get()) == false)
    {
    daxErrorMacro("Incompatible port types");
    return false;
    }

  daxInternals::Graph::vertex_descriptor sourceVertex, sinkVertex;

  // Need to find the vertex for sourceModule and add a new one only if none was
  // found.
  std::pair<daxInternals::VertexIterator, daxInternals::VertexIterator> viter =
      boost::vertices(this->Internals->Connectivity);
  sourceVertex = sinkVertex = *viter.second;
  for (; viter.first != viter.second &&
    (sourceVertex == *viter.second || sinkVertex == *viter.second); ++viter.first)
    {
    if (this->Internals->Connectivity[*viter.first].Module == sourceModule)
      {
      sourceVertex = *viter.first;
      }
    else if (this->Internals->Connectivity[*viter.first].Module == sinkModule)
      {
      sinkVertex = *viter.first;
      }

    }

  if (sourceVertex == *viter.second)
    {
    sourceVertex = boost::add_vertex(this->Internals->Connectivity);
    this->Internals->Connectivity[sourceVertex].Module = sourceModule;
    }
  if (sinkVertex == *viter.second)
    {
    sinkVertex = boost::add_vertex(this->Internals->Connectivity);
    this->Internals->Connectivity[sinkVertex].Module = sinkModule;
    }

  // Now add the egde, unless it already exists.
  daxInternals::EdgeIterator start, end;
  // using boost::tie instead of std::pair to achieve the same effect.
  boost::tie(start, end) = boost::edges(this->Internals->Connectivity);
  daxInternals::Graph::edge_descriptor edge = *end;
  for (; start != end && edge == *end; start++)
    {
    if (this->Internals->Connectivity[*start].ProducerPort == sourcePort &&
      this->Internals->Connectivity[*start].ConsumerPort == sinkPort)
      {
      edge = *start;
      }
    }

  if (edge == *end)
    {
    edge = boost::add_edge(sourceVertex, sinkVertex,
      this->Internals->Connectivity).first;
    this->Internals->Connectivity[edge].ProducerPort = sourcePort;
    this->Internals->Connectivity[edge].ConsumerPort = sinkPort;
    }
  return true;
}

//-----------------------------------------------------------------------------
void daxExecutive2::Reset()
{
  this->Internals->Connectivity = daxInternals::Graph();
}
//daxHeaderString_Kernel

//-----------------------------------------------------------------------------
void daxExecutive2::PrintKernel()
{
  ctemplate::TemplateDictionary dictionary("kernel");
  dictionary.SetValue("dax_array_count", "3");
  
  // 2 inputs, 1 output.
  ctemplate::TemplateDictionary *input0 = dictionary.AddSectionDictionary(
    "dax_input_arrays");
  input0->SetValue("dax_name", "input0");
  input0->SetValue("INDEX", "0");

  ctemplate::TemplateDictionary *input1 = dictionary.AddSectionDictionary(
    "dax_input_arrays");
  input1->SetValue("dax_name", "input1");
  input1->SetValue("INDEX", "1");

  ctemplate::TemplateDictionary *output0 = dictionary.AddSectionDictionary(
    "dax_output_arrays");
  output0->SetValue("dax_name", "output0");
  output0->SetValue("INDEX", "2");

  ctemplate::TemplateDictionary *dax_generators = dictionary.AddSectionDictionary(
    "dax_generators");
  dax_generators->SetValue("dax_name", "1");
  dax_generators->SetValue("dax_invoke", "functor1(work, arrays[0], arrays[1]);");

  dax_generators = dictionary.AddSectionDictionary(
    "dax_generators");
  dax_generators->SetValue("dax_name", "2");
  dax_generators->SetValue("dax_invoke", "functor2(work, arrays[1], arrays[2]);");

  ctemplate::Template* tmpl = ctemplate::Template::StringToTemplate(
    daxHeaderString_Kernel, ctemplate::STRIP_BLANK_LINES);

  std::string result;
  tmpl->Expand(&result, &dictionary);
  cout << result.c_str() << endl;
}
