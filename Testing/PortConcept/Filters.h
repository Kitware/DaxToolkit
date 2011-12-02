#ifndef FILTERS_H
#define FILTERS_H

#include "FilterConnecters.h"

#include "boost/function.hpp"
#include "boost/bind.hpp"

#include <string>
#include <vector>

//------------------------------------------------------------------------------
template < typename ModuleFunction >
class Filter
{
private:
  std::vector< boost::function<void(void)> > Dependencies;
  template< class Base>
  void addDependency(Base &filter)
    {
    Dependencies.push_back(boost::bind(&Base::execute,&filter));
    }
public:

  template <typename TypeA>
  Filter(const FilterConnector<TypeA>& input):
    Module_(input.port())
  {}

  template <typename TypeA, typename TypeB>
  Filter(const FilterConnector<TypeA>& input,
         const FilterConnector<TypeB>& input2):
    Module_(input.port(),input2.port())
  {}

  template <typename TypeA, typename TypeB,typename TypeC>
  Filter(const FilterConnector<TypeA>& input,
         const FilterConnector<TypeB>& input2,
         const FilterConnector<TypeC>& input3):
  Module_(input.port(),input2.port(),input3.port())
  {}

  template <typename TypeA, typename TypeB, typename TypeC, typename TypeD>
  Filter(const FilterConnector<TypeA>& input,
         const FilterConnector<TypeB>& input2,
         const FilterConnector<TypeC>& input3,
         const FilterConnector<TypeD>& input4):
    Module_(input.port(),input2.port(),
            input3.port(),input4.port())
  {}


  template <typename ModuleFunctionA>
  Filter(Filter<ModuleFunctionA>& inputFilter):
    Module_(inputFilter.output().port())
  {

  }

  template <typename ModuleFunctionA, typename ModuleFunctionB>
  Filter(Filter<ModuleFunctionA>& inputFilter, Filter<ModuleFunctionB>& inputFilter2):
    Module_(inputFilter.output().port(),
          inputFilter2.output().port())
  {

  }

  template <typename ModuleFunctionA, typename ModuleFunctionB, typename ModuleFunctionC>
  Filter(Filter<ModuleFunctionA>& inputFilterA, Filter<ModuleFunctionB>& inputFilterB,
         Filter<ModuleFunctionC>& inputFilterC):
    Module_(inputFilterA.output().port(),
          inputFilterB.output().port(),
          inputFilterC.output().port())
  {

  }

  template <typename ModuleFunctionA, typename ModuleFunctionB, typename ModuleFunctionC, typename ModuleFunctionD>
  Filter(Filter<ModuleFunctionA>& inputFilterA, Filter<ModuleFunctionB>& inputFilterB,
          Filter<ModuleFunctionC>& inputFilterC, Filter<ModuleFunctionD>& inputFilterD):
    Module_(inputFilterA.output().port(),
          inputFilterB.output().port(),
          inputFilterC.output().port(),
          inputFilterD.output().port())
  {

  }

  FilterConnector< Filter<ModuleFunction > > output(const int index=0) const
  {
  return FilterConnector< Filter<ModuleFunction > >(this,
                                                    Module_.outputPort(index));
  }

  std::string fieldType(const int index=0) const
  {
    return this->output(index).port().fieldType();
  }

  int size(const int index=0) const
  {
    return this->output(index).port().size();
  }

  void execute()
  {
    std::cout << "starting run" << std::endl;
     Module_.execute();
  }

  ModuleFunction Module_;
};

//------------------------------------------------------------------------------
// Functions that generate Port objects, a potential way around
// having models and making it easier to make filter connections
template<typename T>
FilterConnector<T> pointField(const FilterConnector<T> &fc, const std::string& name)
{
  return FilterConnector<T>(fc,name,field_points());
}

template <typename T>
FilterConnector< Filter<T> > pointField(const Filter<T>& filter, const std::string& name)
{
  return pointField(filter.output(0),name);
}

//------------------------------------------------------------------------------
template <typename T>
FilterConnector<T> cellField(const FilterConnector<T> &fc, const std::string& name )
{
  return FilterConnector<T>(fc,name,field_cells());
}

template <typename T>
FilterConnector< Filter<T> > cellField(const Filter<T>& filter, const std::string& name )
{
  return cellField(filter.outputPort(0),name);
}

//------------------------------------------------------------------------------
template <typename T>
FilterConnector<T> points(const FilterConnector<T> &fc)
{
  return FilterConnector<T>(fc,field_points());
}

template <typename T>
FilterConnector< Filter<T> > points(const Filter<T>& filter)
{
  return points(filter.output(0));
}

//------------------------------------------------------------------------------
template <typename T>
FilterConnector<T> topology(const FilterConnector<T> &fc )
{
  return FilterConnector<T>(fc,field_topology());
}

template <typename T>
FilterConnector< Filter<T> > topology(const Filter<T>& filter)
{
  return topology(filter.output(0));
}

#endif // FILTERS_H
