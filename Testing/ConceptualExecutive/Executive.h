#ifndef EXECUTIVE_H
#define EXECUTIVE_H

#include <vector>
#include <string>

#include "daxTypes.h"

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/graph/topological_sort.hpp>
#include <utility>                   // for std::pair


class Executive
{
private:
  /// Meta-data stored with each vertex in the graph.
  struct VertexProperty
    {

    };

  /// Meta-data stored with each edge in the graph.
  struct EdgeProperty
    {

    };

  typedef boost::adjacency_list<boost::vecS,
          boost::vecS,
          boost::directedS,
          /* Vertex Property*/
          VertexProperty,
          /* Edge Property*/
          EdgeProperty > Graph;

  typedef boost::graph_traits<Graph>::vertex_iterator VertexIterator;
  typedef boost::graph_traits<Graph>::edge_iterator EdgeIterator;
  typedef boost::graph_traits<Graph>::in_edge_iterator InEdgeIterator;
  Graph Connectivity;

public:
  typedef boost::shared_ptr<Executive> ExecutivePtr;
  void run();

  template<typename Worklet, typename ArrayType>
  void connect(ArrayType *type)
  {
  }

  template<typename PrevWorklet, typename CurrentWorklet>
  void connect(const dax::worklets::BaseFieldWorklet<PrevWorklet> &in,
               dax::worklets::BaseFieldWorklet<CurrentWorklet> &vert )
  {

  }

};

#endif // EXECUTIVE_H
