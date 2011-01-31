/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "fncExecutive.h"

// Generated headers.
#include "CoreKernel.cl.h"
#include "CoreMapField.cl.h"
#include "CorePointIterator.cl.h"

// fnc headers
#include "fncModule.h"
#include "fncPort.h"

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

class fncExecutive::fncInternals
{
public:
  /// Meta-data stored with each vertex in the graph.
  struct VertexProperty
    {
    fncModulePtr Module;
    };

  /// Meta-data stored with each edge in the graph.
  struct EdgeProperty
    {
    fncPortPtr ProducerPort;
    fncPortPtr ConsumerPort;
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

namespace fnc
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
fncExecutive::fncExecutive()
{
  this->Internals = new fncInternals();
}

//-----------------------------------------------------------------------------
fncExecutive::~fncExecutive()
{
  delete this->Internals;
  this->Internals = NULL;
}

//-----------------------------------------------------------------------------
bool fncExecutive::Connect(
  fncModulePtr sourceModule, const std::string& sourcename,
  fncModulePtr sinkModule, const std::string& sinkname)
{
  return this->Connect(sourceModule, sourceModule->GetOutputPort(sourcename),
    sinkModule, sinkModule->GetInputPort(sinkname));
}

//-----------------------------------------------------------------------------
bool fncExecutive::Connect(
  fncModulePtr sourceModule, fncPortPtr sourcePort,
  fncModulePtr sinkModule, fncPortPtr sinkPort)
{
  assert(sourceModule && sourcePort && sinkModule && sinkPort);
  assert(sourceModule != sinkModule);

  // * Validate that their types match up.
  if (sinkPort->CanSourceFrom(sourcePort.get()) == false)
    {
    fncErrorMacro("Incompatible port types");
    return false;
    }

  fncInternals::Graph::vertex_descriptor sourceVertex, sinkVertex;

  // Need to find the vertex for sourceModule and add a new one only if none was
  // found.
  std::pair<fncInternals::VertexIterator, fncInternals::VertexIterator> viter =
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
  fncInternals::EdgeIterator start, end;
  // using boost::tie instead of std::pair to achieve the same effect.
  boost::tie(start, end) = boost::edges(this->Internals->Connectivity);
  fncInternals::Graph::edge_descriptor edge = *end;
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
void fncExecutive::Reset()
{
  this->Internals->Connectivity = fncInternals::Graph();
}

//-----------------------------------------------------------------------------
bool fncExecutive::Execute(const fncImageData* input, fncImageData* output)
{
  cout << endl << "--------------------------" << endl;
  cout << "Connectivity Graph: " << endl;
  boost::print_graph(this->Internals->Connectivity);
  cout << "--------------------------" << endl;

  // Every sink becomes a separate kernel.

  // Locate sinks. Sinks are nodes in the dependency graph with in-degree of 0.
  std::vector<fncInternals::Graph::vertex_descriptor> sinks;
  fnc::get_heads(sinks, this->Internals->Connectivity);

  // Now we process each sub-graph rooted at each sink separately, create
  // separate kernels and executing them individually.

  // Maybe using for_each isn't the best thing here since I want to break on
  // error. I am just letting myself get a little carried away with boost::bind
  // and std::for_each temporarily :).
  std::for_each(sinks.begin(), sinks.end(),
    boost::bind(&fncExecutive::ExecuteOnce<fncInternals::Graph, fncImageData,
      fncImageData>,
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
    fncModulePtr module = graph[vertex].Module;
    fncModule::Types module_type = module->GetType();

    switch (module_type)
      {
    case fncModule::map_field:
        {
        // this operates on the same field as passed on in the input.
        StackItem top = this->CommandStack.back();
        assert(top.IsOperator());
        this->CommandStack.pop_back();
        this->CommandStack.push_back(StackItem(vertex));
        this->CommandStack.push_back(top);
        }
      break;

    case fncModule::map_topology_down:
      cout << "TODO" << endl;
      abort();
      break;

    case fncModule::map_topology_up:
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
  CommandStackEntity(fncPort::Types operator_)
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
    fncPort::Types Operator;
    };
};

namespace fnc
{
  std::string fncSubstituteKeywords(
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
      typename Graph::vertex_descriptor head, const Graph& graph)
      {
      // The algorithm we use is as follows:
      // We treat the problem like solving a mathematical expression using operand
      // and operator stacks. Operators are iterators. And all operators are
      // unary. Operands are modules(functors). Consecutive operators of the same
      // type can be combined into a single iteration.

      typedef std::vector<CommandStackEntity<Graph> > CommandStack;
      CommandStack command_stack;

      // Now based on our data-input, we push the first operator on the stack.
      // We are assuming image-data with point scalars to begin with.
      command_stack.push_back(CommandStackEntity<Graph>(fncPort::point_array));

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
          case fncPort::point_array:
              {
              opencl_kernel = fncSubstituteKeywords(
                fncHeaderString_CorePointIterator, keywords);
              }
            break;

          default:
            cout << __LINE__ << " : TODO" << endl;
            }
          operator_index++;
          }
        else // if (item.IsOperand())
          {
          fncModule::Types module_type = graph[item.Operand].Module->GetType();
          switch (module_type)
            {
          case fncModule::map_field:
            keywords["module_name"] = graph[item.Operand].Module->GetModuleName();
            keywords["vertexid"] = (boost::format("%1%") % item.Operand).str();
            opencl_kernel = fncSubstituteKeywords(
              fncHeaderString_CoreMapField, keywords) + opencl_kernel;
            }
          }
        }
      keywords["topology_opaque_pointer"] = "opaque_data_pointer";
      keywords["input_data_handle"] = "input_point_array";
      keywords["output_data_handle"] = "output_point_array";
      keywords["body"] = opencl_kernel;
      opencl_kernel = fncSubstituteKeywords(fncHeaderString_CoreKernel, keywords);
      cout << "================================================" <<endl;
      cout << opencl_kernel.c_str() << endl;
      return opencl_kernel;
      }
}

//-----------------------------------------------------------------------------
template <class Graph, class InputDataType, class OutputDataType>
bool fncExecutive::ExecuteOnce(
  typename Graph::vertex_descriptor head, const Graph& graph,
  const InputDataType* input, OutputDataType* output)
{
  // FIXME: now sure how to dfs over a subgraph starting with head, so for now
  // we assume there's only 1 connected graph in "graph".
  cout << "Execute sub-graph: " << head << endl;

  // First generate the kernel code.
  std::string kernel = fnc::GenerateKernel(head, graph);

  // Now we should invoke the kernel using opencl setting up data arrays etc
  // etc.
  return false;
}
