/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "fncExecutive.h"

#include "fncPort.h"
#include "fncModule.h"

#include <assert.h>
#include <algorithm>
#include <boost/config.hpp>
#include <boost/bind.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/graph/graph_utility.hpp>
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
bool fncExecutive::Execute()
{
  cout << endl << "--------------------------" << endl;
  cout << "Connectivity Graph: " << endl;
  boost::print_graph(this->Internals->Connectivity);
  cout << "--------------------------" << endl;

  typedef boost::reverse_graph<fncInternals::Graph> dependencyGraphType;
  dependencyGraphType dependencyGraph(this->Internals->Connectivity);
  cout << endl << "Dependency Graph: " << endl;
  boost::print_graph(dependencyGraph);
  cout << "--------------------------" << endl << endl;

  // Every sink becomes a separate kernel.

  // Locate sinks. Sinks are nodes in the dependency graph with in-degree of 0.
  std::vector<fncInternals::Graph::vertex_descriptor> sinks;
  fnc::get_heads(sinks, dependencyGraph);

  // Now we process each sub-graph rooted at each sink separately, create
  // separate kernels and executing them individually.

  // Maybe using for_each isn't the best thing here since I want to break on
  // error :).
  std::for_each(sinks.begin(), sinks.end(),
    boost::bind(&fncExecutive::Execute<dependencyGraphType>,
      this, _1, dependencyGraph));
  return false;
}

//-----------------------------------------------------------------------------
template <class Graph>
bool fncExecutive::Execute(
  typename Graph::vertex_descriptor head, const Graph& graph)
{
  return false;
}
