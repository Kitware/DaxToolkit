/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxExecutive2.h"

// Generated headers.
#include "Kernel.tmpl.h"
#include "KernelGetArray.tmpl.h"
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
#include <map>
#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>
#include <boost/config.hpp>
#include <boost/format.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/graph/topological_sort.hpp>
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
  typedef boost::graph_traits<Graph>::in_edge_iterator InEdgeIterator;
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
std::string daxExecutive2::GetKernel()
{
  //cout << endl << "--------------------------" << endl;
  //cout << "Connectivity Graph: " << endl;
  //boost::print_graph(this->Internals->Connectivity);
  //cout << "--------------------------" << endl;

  ctemplate::TemplateDictionary dictionaryKernel("kernel");
  ctemplate::TemplateDictionary dictionaryGetArray("get_array");

  ctemplate::Template* getArrayTemplate = ctemplate::Template::StringToTemplate(
    daxHeaderString_KernelGetArray, ctemplate::STRIP_BLANK_LINES);

  // We can just do a topological sort and then invoke the kernel.
  std::vector<daxInternals::Graph::vertex_descriptor> sorted_vertices;

  boost::reverse_graph<daxInternals::Graph> rev_connectivity(this->Internals->Connectivity);
  boost::topological_sort(rev_connectivity, std::back_inserter(sorted_vertices));

  // every port gets an array index.
  std::map<daxPort*, int> index_map;
  std::map<daxPort*, int> global_outputs_map;

  int num_of_arrays = 0;
  int input_array_count = 0;
  for (size_t cc=0; cc < sorted_vertices.size(); cc++)
    {
    daxModulePtr module =
      this->Internals->Connectivity[sorted_vertices[cc]].Module;
    size_t num_input_ports = module->GetNumberOfInputs();
    size_t num_output_ports = module->GetNumberOfOutputs();

    dictionaryGetArray.SetValue("dax_name", module->GetModuleName());
    std::string result;
    getArrayTemplate->Expand(&result, &dictionaryGetArray);
    dictionaryKernel.SetValueAndShowSection(
      "dax_get_array_kernel", result, "dax_get_array_kernels");

    ctemplate::TemplateDictionary* moduleDict =
      dictionaryGetArray.AddSectionDictionary("dax_generators");
    moduleDict->SetValue("dax_id", (boost::format("%1%") % (cc+1)).str());
    moduleDict->SetValue("dax_name", module->GetModuleName());

    // name module's out-arrays. Every out-array gets a unique name.
    for (size_t portno = 0; portno < num_output_ports; portno++)
      {
      int index = num_of_arrays++;
      index_map[module->GetOutputPort(portno).get()] = index;
      // Initially, we'll assume any output-array not consumed by another
      // functor is a global output array. To make it easier to determine those
      // arrays, we put the output ports in another map. As they are used, we
      // remove them from the map. Whatever remains in the map are global output
      // arrays.
      global_outputs_map[module->GetOutputPort(portno).get()] = index;

      ctemplate::TemplateDictionary* arraysDict =
        dictionaryKernel.AddSectionDictionary("dax_generated_arrays");
      arraysDict->SetValue("dax_index", (boost::format("%1%")%index).str());
      arraysDict->SetValue("dax_generator_id",
        (boost::format("%1%") % (cc+1)).str());
      }

    // Now to determine what are the names of the input arrays. There are two
    // possibilities: they are either global arrays or output-arrays from other
    // modules.
    daxInternals::InEdgeIterator start_in, end_in;
    boost::tie(start_in, end_in) = boost::in_edges(sorted_vertices[cc],
      this->Internals->Connectivity);
    for (; start_in != end_in; start_in++)
      {
      daxPort* producer =
        this->Internals->Connectivity[*start_in].ProducerPort.get();
      daxPort* consumer =
        this->Internals->Connectivity[*start_in].ConsumerPort.get();
      assert(index_map.find(consumer) == index_map.end());
      if (index_map.find(producer) != index_map.end())
        {
        // producer is consumed, so no longer a global output array.
        global_outputs_map.erase(producer);

        index_map[consumer] = index_map[producer];
        }
      }

    for (size_t portno=0; portno < num_input_ports; portno++)
      {
      daxPort* port = module->GetInputPort(portno).get();
      if (index_map.find(port) == index_map.end())
        {
        // this is a global-input
        int array_index = num_of_arrays++;
        index_map[port] = array_index;

        ctemplate::TemplateDictionary* input_array_dict =
          dictionaryKernel.AddSectionDictionary("dax_input_arrays");
        input_array_dict->SetValue("dax_name",
          (boost::format("input_array_%1%")%(input_array_count++)).str());
        input_array_dict->SetValue("dax_index",
          (boost::format("%1%")%(array_index)).str());
        }
      }

    for (size_t portno=0; portno < num_input_ports; portno++)
      {
      daxPort* port = module->GetInputPort(portno).get();
      assert(index_map.find(port) != index_map.end());
      moduleDict->SetValueAndShowSection("dax_array_index",
        (boost::format("%1%") % index_map[port] ).str(), "dax_args");
      }

    for (size_t portno=0; portno < num_output_ports; portno++)
      {
      daxPort* port = module->GetOutputPort(portno).get();
      assert(index_map.find(port) != index_map.end());
      moduleDict->SetValueAndShowSection("dax_array_index",
        (boost::format("%1%") % index_map[port] ).str(), "dax_args");
      }
    }

  int output_array_count = 0;
  for (std::map<daxPort*, int>::iterator iter = global_outputs_map.begin();
    iter != global_outputs_map.end(); ++iter)
    {
    ctemplate::TemplateDictionary* output_array_dict =
      dictionaryKernel.AddSectionDictionary("dax_output_arrays");
    output_array_dict->SetValue("dax_name",
      (boost::format("output_array_%1%")%(output_array_count++)).str());
    output_array_dict->SetValue("dax_index",
      (boost::format("%1%")%iter->second).str());
    }

  std::string result;
  dictionaryGetArray.SetValue("dax_name", "__final__");
  getArrayTemplate->Expand(&result, &dictionaryGetArray);
  dictionaryKernel.SetValueAndShowSection(
    "dax_get_array_kernel", result, "dax_get_array_kernels");

  dictionaryKernel.SetValue("dax_array_count",
    (boost::format("%1%")%num_of_arrays).str());
  //dictionaryKernel.Dump();

  ctemplate::Template* tmpl = ctemplate::Template::StringToTemplate(
    daxHeaderString_Kernel, ctemplate::STRIP_BLANK_LINES);

  std::string t_result;
  tmpl->Expand(&t_result, &dictionaryKernel);
  return t_result;
}

//-----------------------------------------------------------------------------
void daxExecutive2::PrintKernel()
{
  std::string result = this->GetKernel();
  cout << result.c_str() << endl;
}

//-----------------------------------------------------------------------------
template <class Graph>
bool daxExecutive2::ExecuteOnce(
  typename Graph::vertex_descriptor head, const Graph& graph) const
{
  // FIXME: now sure how to dfs over a subgraph starting with head, so for now
  // we assume there's only 1 connected graph in "graph".
  //cout << "Execute sub-graph: " << head << endl;

  std::vector<std::string> functor_codes;

  // First generate the kernel code.
  //std::string kernel = dax::GenerateKernel(head, graph, functor_codes);
  return false;
}

//-----------------------------------------------------------------------------
void daxExecutive2PrintKernel()
{
  ctemplate::TemplateDictionary dictionary("kernel");
  dictionary.SetValue("dax_array_count", "3");

  // 2 inputs, 1 output.
  ctemplate::TemplateDictionary *input0 = dictionary.AddSectionDictionary(
    "dax_input_arrays");
  input0->SetValue("dax_name", "input0");
  input0->SetValue("dax_index", "0");

  ctemplate::TemplateDictionary *input1 = dictionary.AddSectionDictionary(
    "dax_input_arrays");
  input1->SetValue("dax_name", "input1");
  input1->SetValue("dax_index", "1");

  ctemplate::TemplateDictionary *output0 = dictionary.AddSectionDictionary(
    "dax_output_arrays");
  output0->SetValue("dax_name", "output0");
  output0->SetValue("dax_index", "2");

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
