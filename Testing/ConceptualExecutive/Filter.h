#ifndef FILTER_H
#define FILTER_H

#include "DataSet.h"
#include "Executive.h"

#include "boost/function.hpp"
#include "boost/bind.hpp"
#include <vector>

//------------------------------------------------------------------------------
template < typename Worklet >
class Filter
{
public:
  typedef typename Worklet::InputType InputType;
  typedef typename Worklet::OutputType OutputType;

private:
  typedef boost::function<void (Executive& exec)> ExecutiveFuncSignature;

  //dataset can be changed but we can't change the pointer location
  std::vector< ExecutiveFuncSignature > Dependencies;

  dax::HostArray<InputType>* InputArray;
  dax::HostArray<OutputType> OutputArray;
  DataSet *Data;

public:
  Filter(DataSet* inputData, dax::HostArray<InputType> *t):
    Data(inputData),
    InputArray(t)
  {}

  template<typename T>
  Filter(const Filter<T>& input):
    Data(input.Data),
    InputArray(input.OutputArray)
  {
    Dependencies.push_back(
          boost::bind(&Filter<T>::addDependenciesToExecutive,&input,_1));
  }

  void run() const
  {
    Executive exec;
    this->addDependenciesToExecutive(exec);
    exec.run();
  }

  void addDependenciesToExecutive(Executive& exec) const
  {
    //recursively do a depth first walk up the tree
    //so that we properly call execute from the
    //top of the pipeline down correctly
    std::vector< ExecutiveFuncSignature >::const_iterator it;
    for(it=Dependencies.begin();it!=Dependencies.end();++it)
      {
      //call my input filters execute function
      (*it)(exec);
      }
    exec.add<Worklet>();
  }
};

#endif // FILTER_H
