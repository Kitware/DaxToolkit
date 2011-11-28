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
public:
  Filter(dax::HostArray<InputType> *t):
  InputArray(t)
  {
    Dependencies.push_back(
          boost::bind(&Filter<Worklet>::addDataToExecutive,boost::ref(*this),_1));
  }

  //constructor when connecting a filter to a different type of filter
  template<typename T>
  Filter(const Filter<T>& input):
    InputArray(NULL)
  {
    Dependencies.push_back(
          boost::bind(&Filter<T>::addDependenciesToExecutive,boost::ref(input),_1));
  }

  //constructor when connecting a filter to an identical filter
  //needed as the T templated constructor doesn't work on those
  Filter(const Filter<Worklet>& input):
    InputArray(NULL)
  {
    Dependencies.push_back(
          boost::bind(&Filter<Worklet>::addDependenciesToExecutive,boost::ref(input),_1));
  }

  void run() const
  {
    Executive exec;
    this->addDependenciesToExecutive(exec);
    exec.run();
  }

  void addDataToExecutive(Executive& exec) const
  {
    exec.addData< dax::HostArray<InputType> >( this->InputArray );
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
