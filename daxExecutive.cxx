/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxExecutive.h"

// Generated headers.
#include "CoreKernel.cl.h"
#include "CoreMapField.cl.h"
#include "CorePointIterator.cl.h"

// dax headers
#include "daxImageData.h"
#include "daxModule.h"
#include "daxOptions.h"
#include "daxPort.h"

// OpenCL headers
#ifdef FNC_ENABLE_OPENCL
//# define __CL_ENABLE_EXCEPTIONS
# include <CL/cl.hpp>
#endif

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

class daxExecutive::daxInternals
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
daxExecutive::daxExecutive()
{
  this->Internals = new daxInternals();
}

//-----------------------------------------------------------------------------
daxExecutive::~daxExecutive()
{
  delete this->Internals;
  this->Internals = NULL;
}

//-----------------------------------------------------------------------------
bool daxExecutive::Connect(
  const daxModulePtr sourceModule, const std::string& sourcename,
  const daxModulePtr sinkModule, const std::string& sinkname)
{
  return this->Connect(sourceModule, sourceModule->GetOutputPort(sourcename),
    sinkModule, sinkModule->GetInputPort(sinkname));
}

//-----------------------------------------------------------------------------
bool daxExecutive::Connect(
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
void daxExecutive::Reset()
{
  this->Internals->Connectivity = daxInternals::Graph();
}

//-----------------------------------------------------------------------------
bool daxExecutive::Execute(
  const daxImageData* input, daxImageData* output) const
{
  cout << endl << "--------------------------" << endl;
  cout << "Connectivity Graph: " << endl;
  boost::print_graph(this->Internals->Connectivity);
  cout << "--------------------------" << endl;

  // Every sink becomes a separate kernel.

  // Locate sinks. Sinks are nodes in the dependency graph with in-degree of 0.
  std::vector<daxInternals::Graph::vertex_descriptor> sinks;
  dax::get_heads(sinks, this->Internals->Connectivity);

  // Now we process each sub-graph rooted at each sink separately, create
  // separate kernels and executing them individually.

  // Maybe using for_each isn't the best thing here since I want to break on
  // error. I am just letting myself get a little carried away with boost::bind
  // and std::for_each temporarily :).
  std::for_each(sinks.begin(), sinks.end(),
    boost::bind(&daxExecutive::ExecuteOnce<daxInternals::Graph, daxImageData,
      daxImageData>,
      this, _1, this->Internals->Connectivity, input, output));
  return false;
}

template <class Graph, class Stack>
class DFSVisitor : public boost::default_dfs_visitor
{
  typedef typename boost::graph_traits<Graph>::edge_descriptor edge_descriptor;
  typedef typename boost::graph_traits<Graph>::vertex_descriptor
    vertex_descriptor;
  typedef typename Stack::value_type StackItem;
  Stack &CommandStack;
  void operator=(const DFSVisitor&); // not implemented.
public:
  DFSVisitor(Stack& stack): CommandStack(stack) { }
  void tree_edge(edge_descriptor edge, const Graph& graph)
    {
    this->handle_vertex(boost::target(edge, graph), graph);
    }

  void start_vertex(vertex_descriptor vertex, const Graph& graph)
    {
    this->handle_vertex(vertex, graph);
    }

  void finish_vertex(vertex_descriptor vertex, const Graph& graph)
    {
    (void)vertex;
    (void)graph;
    }
private:
  void handle_vertex(vertex_descriptor vertex, const Graph& graph)
    {
    cout << "Handle: " << vertex << endl;
    assert(this->CommandStack.rbegin() != this->CommandStack.rend());
    // * Determine the type of the module.
    daxModulePtr module = graph[vertex].Module;
    daxModule::Types module_type = module->GetType();

    switch (module_type)
      {
    case daxModule::map_field:
        {
        // this operates on the same field as passed on in the input.
        StackItem top = this->CommandStack.back();
        assert(top.IsOperator());
        this->CommandStack.pop_back();
        this->CommandStack.push_back(StackItem(vertex));
        this->CommandStack.push_back(top);
        }
      break;

    case daxModule::map_topology_down:
      cout << "TODO" << endl;
      abort();
      break;

    case daxModule::map_topology_up:
      cout << "TODO" << endl;
      abort();
      break;

    default:
      abort();
      }
    }
};

template <class Graph>
class CommandStackEntity
{
public:
  enum Types
    {
    OPERAND,
    OPERATOR
    };
  CommandStackEntity(typename Graph::vertex_descriptor vertex)
    {
    this->Type = OPERAND;
    this->Operand = vertex;
    }
  CommandStackEntity(daxPort::Types operator_)
    {
    this->Type = OPERATOR;
    this->Operator = operator_;
    }
  bool IsOperator() const { return this->Type == OPERATOR; }
  bool IsOperand() const { return this->Type == OPERAND; }

  Types Type;
  union
    {
    typename Graph::vertex_descriptor Operand;
    daxPort::Types Operator;
    };
};

namespace dax
{
  std::string daxSubstituteKeywords(
    const char* source, const std::map<std::string, std::string>& keywords)
    {
    std::string result = source;
    for (std::map<std::string, std::string>::const_iterator
      iter = keywords.begin(); iter != keywords.end(); ++iter)
      {
      boost::replace_all(result, "$" + iter->first + "$", iter->second);
      }
    return result;
    }

  //-----------------------------------------------------------------------------
  template <class Graph>
    std::string GenerateKernel(
      typename Graph::vertex_descriptor head, const Graph& graph,
      std::vector<std::string>& kernels)
      {
      std::map<std::string, std::string> kernel_map;
      // The algorithm we use is as follows:
      // We treat the problem like solving a mathematical expression using operand
      // and operator stacks. Operators are iterators. And all operators are
      // unary. Operands are modules(functors). Consecutive operators of the same
      // type can be combined into a single iteration.

      typedef std::vector<CommandStackEntity<Graph> > CommandStack;
      CommandStack command_stack;

      // Now based on our data-input, we push the first operator on the stack.
      // We are assuming image-data with point scalars to begin with.
      command_stack.push_back(CommandStackEntity<Graph>(daxPort::point_array));

      DFSVisitor<Graph, CommandStack> visitor(command_stack);
      // Do a DFS-visit starting at the head. We are assuming no fan-ins or
      // fan-outs.
      boost::depth_first_search(graph, boost::visitor(visitor).root_vertex(head));

      std::string opencl_kernel;
      std::map<std::string, std::string> keywords;
      size_t operator_index = 0;
      for (typename CommandStack::iterator iter = command_stack.begin();
        iter != command_stack.end(); ++iter)
        {
        const typename CommandStack::value_type &item = *iter;
        if (item.IsOperator())
          {
          keywords["index"] = (boost::format("%1%") % operator_index).str();
          keywords["body"] = opencl_kernel;
          switch (item.Operator)
            {
          case daxPort::point_array:
              {
              opencl_kernel = daxSubstituteKeywords(
                daxHeaderString_CorePointIterator, keywords);
              }
            break;

          default:
            cout << __LINE__ << " : TODO" << endl;
            }
          operator_index++;
          }
        else // if (item.IsOperand())
          {
          daxModule::Types module_type = graph[item.Operand].Module->GetType();
          std::string module_name = graph[item.Operand].Module->GetModuleName();
          kernel_map[module_name] = graph[item.Operand].Module->GetFunctorCode();
          switch (module_type)
            {
          case daxModule::map_field:
            keywords["module_name"] = module_name;
            keywords["vertexid"] = (boost::format("%1%") % item.Operand).str();
            opencl_kernel = daxSubstituteKeywords(
              daxHeaderString_CoreMapField, keywords) + opencl_kernel;
          default:
            cout << __LINE__ << ": TODO" << endl;
            }
          }
        }
      keywords["topology_opaque_pointer"] = "opaque_data_pointer";
      keywords["input_data_handle"] = "inputHandle";
      keywords["output_data_handle"] = "outputHandle";
      keywords["body"] = opencl_kernel;
      opencl_kernel = daxSubstituteKeywords(daxHeaderString_CoreKernel, keywords);
      cout << "================================================" <<endl;
      cout << opencl_kernel.c_str() << endl;

      for (std::map<std::string, std::string>::iterator
        iter = kernel_map.begin(); iter != kernel_map.end(); ++iter)
        {
        kernels.push_back(iter->second);
        }
      return opencl_kernel;
      }
}

#define RETURN_ON_ERROR(err, msg) \
  {\
  if (err != CL_SUCCESS)\
    {\
    cerr << __FILE__<<":"<<__LINE__ << endl<< "ERROR:  Failed to " << msg << endl;\
    return false;\
    }\
  }

//-----------------------------------------------------------------------------
template <class Graph, class InputDataType, class OutputDataType>
bool daxExecutive::ExecuteOnce(
  typename Graph::vertex_descriptor head, const Graph& graph,
  const InputDataType* input, OutputDataType* output) const
{
  // FIXME: now sure how to dfs over a subgraph starting with head, so for now
  // we assume there's only 1 connected graph in "graph".
  cout << "Execute sub-graph: " << head << endl;

  std::vector<std::string> functor_codes;

  // First generate the kernel code.
  std::string kernel = dax::GenerateKernel(head, graph, functor_codes);

#ifndef FNC_ENABLE_OPENCL

  cerr <<
    "You compiled without OpenCL support. So can't really execute "
    "anything. Here's the generated kernel. Have fun!" << endl;
  cerr << kernel.c_str() << endl;
  return false;

#else

  // Now we should invoke the kernel using opencl setting up data arrays etc
  // etc.
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0)
    {
    cout << "No OpenCL capable platforms located." << endl;
    return false;
    }

  cl_int err_code;
  try
    {
    cl::Context context(CL_DEVICE_TYPE_GPU, NULL, NULL, NULL, &err_code);
    RETURN_ON_ERROR(err_code, "create GPU Context");

    // Query devices.
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (devices.size()==0)
      {
      cout << "No OpenGL device located." << endl;
      return -1;
      }

    // Allocate input and output buffers.
    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      daxReadableDataTraits<InputDataType>::GetDataSize("mm..not sure", input),
      const_cast<void*>(
        daxReadableDataTraits<InputDataType>::GetDataPointer("mm..not sure",
          input)),
      /* for data sources that have multiple arrays, we need some mechanism of
       * letting the invoker pick what arrays we are operating on */
      &err_code);
    RETURN_ON_ERROR(err_code, "upload input data");

    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
      daxReadableDataTraits<OutputDataType>::GetDataSize("mm..not sure", output),
      const_cast<void*>(
        daxReadableDataTraits<OutputDataType>::GetDataPointer(
          "mm..not sure", output)),
      &err_code);
    RETURN_ON_ERROR(err_code, "create output_buffer");

    std::string input_data_code = daxOpenCLTraits<InputDataType>::GetCode();
    // Now compile the code.
    cl::Program::Sources sources;
    sources.push_back(std::make_pair(input_data_code.c_str(),
        input_data_code.size()));

    // push functor codes.
    for (size_t cc=0; cc < functor_codes.size(); cc++)
      {
      sources.push_back(std::make_pair(functor_codes[cc].c_str(),
          functor_codes[cc].size()));
      }
    sources.push_back(std::make_pair(kernel.c_str(), kernel.size()));

    // Build the code.
    cl::Program program (context, sources);
    err_code = program.build(devices);
    if (err_code != CL_SUCCESS)
      {
      std::string info;
      program.getBuildInfo(devices[0],
        CL_PROGRAM_BUILD_LOG, &info);
      cout << info.c_str() << endl;
      }
    RETURN_ON_ERROR(err_code, "compile the kernel.");

    // * determine the shape of the kernel invocation.

    // Kernel-shape will be decided by the output data type and the "head"
    // functor module.
    // For now, we simply invoke the kernel per item.

    // * pass arguments to the kernel
    // * invoke the kernel
    // * read back the result
    }
#ifdef __CL_ENABLE_EXCEPTIONS
  catch (cl::Error error)
    {
    cout << error.what() << "(" << error.err() << ")" << endl;
    }
#else
  catch (...)
    {
    cerr << "EXCEPTION"<< endl;
    return false;
    }
#endif

  cout << "So far so good" << endl;
  return false;
#endif
}
