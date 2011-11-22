#ifndef FILTER_H
#define FILTER_H

#include "Port.h"

//------------------------------------------------------------------------------
template < typename Functor >
class Filter1
{
public:
  typedef Functor Function;

  Filter1(const Port& input):
    Funct(input)
  {}

  template <typename FilterA>
  Filter1(FilterA& inputFilter):
    Funct(inputFilter.OutputPort( ))
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

//------------------------------------------------------------------------------
template < typename Functor >
class Filter2
{
public:
  typedef Functor Function;

  Filter2(const Port& input, const Port& input2):
    Funct(input,input2)
  {}

  template <typename FilterA, typename FilterB>
  Filter2(FilterA& inputFilter, FilterB& inputFilter2):
    Funct(inputFilter.OutputPort( ),
          inputFilter2.OutputPort( ))
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

//------------------------------------------------------------------------------
template < typename Functor >
class Filter3
{
public:
  typedef Functor Function;

  Filter3(const Port& input, const Port& input2, const Port& input3):
    Funct(input,input2,input3)
  {}

  template <typename FilterA, typename FilterB, typename FilterC>
  Filter3(FilterA& inputFilterA, FilterB& inputFilterB,
         FilterC& inputFilterC):
    Funct(inputFilterA.OutputPort( ),
          inputFilterB.OutputPort( ),
          inputFilterC.OutputPort( ))
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

//------------------------------------------------------------------------------
template < typename Functor >
class Filter4
{
public:
  typedef Functor Function;

  Filter4(const Port& input, const Port& input2, const Port& input3,
         const Port& input4):
    Funct(input,input2,input3,input4)
  {}

  template <typename FilterA, typename FilterB, typename FilterC, typename FilterD>
  Filter4(FilterA& inputFilterA, FilterB& inputFilterB,
          FilterC& inputFilterC, FilterD& inputFilterD):
    Funct(inputFilterA.OutputPort( ),
          inputFilterB.OutputPort( ),
          inputFilterC.OutputPort( ),
          inputFilterD.OutputPort( ))
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
