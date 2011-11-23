#ifndef FILTER_H
#define FILTER_H

#include "Port.h"
#include "Module.h"


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

  Filter(Port* inputPort)
  {}

  template<typename T>
  Filter(const Filter<T>& input)
  {
    Dependencies.push_back(boost::bind(&Filter<T>::run,input));
  }

  void run()
  {
    this->executeDependencies();
    //execute<Worklet>(Input,Output);
  }

protected:
  void executeDependencies()
  {
    //recursively do a depth first walk up the tree
    //so that we properly call execute from the
    //top of the pipeline down correctly
    std::vector< boost::function<void(void)> >::iterator it;
    for(it=Dependencies.begin();it!=Dependencies.end();++it)
      {
      //call my input filters execute function
      (*it)();
      }

  }

  //we really need iterators as we can specity the
  //input and output types as mere iterators
  //todo: wednesday is iterators
//  dax::basic_iterator input;
//  dax::basic_iterator output;

  //dataset can be changed but we can't change the pointer location
  std::vector< boost::function<void(void)> > Dependencies;
};

#endif // FILTER_H
