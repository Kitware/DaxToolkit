/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_Filter_h
#define __dax_cont_Filter_h

#include <dax/Types.h>

#include "boost/function.hpp"
#include "boost/bind.hpp"

#include <string>
#include <vector>

namespace dax { namespace cont {
template<typename T>
class Filter
{
};
} }

namespace dax { namespace cont { namespace internal {

template<typename Derived>
class FilterBase
{
public:
  template< class Base>
  void addDependency(Base &filter)
    {
    Dependencies.push_back(boost::bind(&Base::execute,&filter));
    }

  FilterBase():AlreadyComputed(false){}

  void execute()
    {
    if(!this->AlreadyComputed)
      {
      this->executeDependencies();
      static_cast<Derived*>(this)->compute();
      this->AlreadyComputed = true;
      }
    }

protected:
  std::vector< boost::function<void(void)> > Dependencies;
  bool AlreadyComputed;

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

};

} } }

#endif
