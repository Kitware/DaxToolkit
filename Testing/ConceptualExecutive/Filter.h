#ifndef FILTER_H
#define FILTER_H

#include "DataSet.h"
#include "Module.h"

//------------------------------------------------------------------------------
template < typename Worklet >
class Filter
{
public:
  typedef typename Worklet::InputType InputType;
  typedef typename Worklet::OutputType OutputType;

  template<typename T>
  Filter(const Filter<T>& input)
  {}

  template<typename T>
  Filter(Filter<T>* input)
  {}


  DataSet* DataSet_;
  Module Module_;
};

#endif // FILTER_H
