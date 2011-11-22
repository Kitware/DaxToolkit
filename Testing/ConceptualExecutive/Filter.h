#ifndef FILTER_H
#define FILTER_H


#include "Port.h"

//------------------------------------------------------------------------------
template < typename Functor >
class Filter
{
public:
  typedef Functor Function;

  Filter(const Port& input):
    Funct(input)
  {}

  Filter(const Port& input, const Port& input2):
    Funct(input,input2)
  {}

  Filter(const Port& input, const Port& input2, const Port& input3):
    Funct(input,input2,input3)
  {}

  Filter(const Port& input, const Port& input2, const Port& input3,
         const Port& input4):
    Funct(input,input2,input3,input4)
  {}


  template <typename FunctorA>
  Filter(Filter<FunctorA>& inputFilter):
    Funct(inputFilter.Funct.outputPort())
  {

  }

  template <typename FunctorA, typename FunctorB>
  Filter(Filter<FunctorA>& inputFilter, Filter<FunctorB>& inputFilter2):
    Funct(inputFilter.Funct.outputPort(),
          inputFilter2.Funct.outputPort())
  {

  }

  template <typename FunctorA, typename FunctorB, typename FunctorC>
  Filter(Filter<FunctorA>& inputFilterA, Filter<FunctorB>& inputFilterB,
         Filter<FunctorC>& inputFilterC):
    Funct(inputFilterA.Funct.outputPort(),
          inputFilterB.Funct.outputPort(),
          inputFilterC.Funct.outputPort())
  {

  }

  template <typename FunctorA, typename FunctorB, typename FunctorC, typename FunctorD>
  Filter(Filter<FunctorA>& inputFilterA, Filter<FunctorB>& inputFilterB,
          Filter<FunctorC>& inputFilterC, Filter<FunctorD>& inputFilterD):
    Funct(inputFilterA.Funct.outputPort(),
          inputFilterB.Funct.outputPort(),
          inputFilterC.Funct.outputPort(),
          inputFilterD.Funct.outputPort())
  {

  }

  const Port& OutputPort( ) const
  {
    return Funct.outputPort();
  }

  int size()
  {
    return Funct.outputPort().size();
  }

  std::string fieldType() const
  {
    return Funct.outputPort().fieldType();
  }

  std::string isMergeable() const
  {
    return Funct.isMergeable() ? "True":"False";
  }

  Function Funct;
};

#endif // FILTER_H
