/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "fncExecutive.h"

#include "fncPort.h"
#include "fncModule.h"

#include <assert.h>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <utility>                   // for std::pair

class fncExecutive::fncInternals
{
public:
  struct VertexProperty
    {
    fncModulePtr Module;
    };

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
  return false;
}
