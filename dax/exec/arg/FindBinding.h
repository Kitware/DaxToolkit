//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_exec_arg_Bind_h
#define __dax_exec_arg_Bind_h
#if defined(DAX_DOXYGEN_ONLY)

#else // !defined(DAX_DOXYGEN_ONLY)

#include <dax/cont/arg/Field.h>
#include <dax/cont/arg/Topology.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/KeyGroup.h>
#include <dax/cont/sig/ReductionCount.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/sig/VisitIndex.h>
#include <dax/cont/sig/WorkId.h>

#include <dax/exec/arg/BindCellPoints.h>
#include <dax/exec/arg/BindCellTag.h>
#include <dax/exec/arg/BindDirect.h>
#include <dax/exec/arg/BindPermutedCellField.h>
#include <dax/exec/arg/BindWorkId.h>
#include <dax/exec/arg/BindKeyGroup.h>
#include <dax/Types.h>

#include <boost/function_types/components.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>


namespace dax { namespace exec { namespace arg {

//bind returns a specific execution and control signature binding
//only specialized for bindings that refer to a control side argument
template <typename DomainType,
          typename Tags,
          typename Invocation,
          int N>
class BindArg
{
public:
  typedef BindDirect<Invocation, N> type;
};

//specialize on Topology to pure arg binding, which is a BindCellTag mapping
template<typename Tags,
         typename Invocation,
         int N>
class BindArg<dax::cont::sig::Cell,
              dax::cont::arg::Topology(Tags),
              Invocation,
              N>
{
public:
  typedef BindCellTag<Invocation, N> type;
};

//specialize on arg to field mapping, with Field(Point) being understood
//as a specialization of the default bind direct behavior
template<typename Tags,
         typename Invocation,
         int N>
class BindArg<dax::cont::sig::Cell,
              dax::cont::arg::Field(Tags),
              Invocation,
              N>
{
public:
  typedef typename boost::mpl::if_<
    typename Tags::template Has<dax::cont::sig::Point>,
    BindCellPoints<Invocation, N>,
    BindDirect<Invocation, N>
    >::type type;
};

//specialize on arg to field mapping when the cells are being permuted
template<typename Tags,
         typename Invocation,
         int N>
class BindArg<dax::cont::sig::PermutedCell,
              dax::cont::arg::Field(Tags),
              Invocation,
              N>
{
public:
  typedef typename boost::mpl::if_<
      typename Tags::template Has<dax::cont::sig::Point>,
      BindCellPoints<Invocation, N>,
      typename boost::mpl::if_<
          typename Tags::template Has<dax::cont::sig::Out>,
          BindDirect<Invocation, N>,
          BindPermutedCellField<Invocation, N> >::type>::type type;

};

template <typename Invocation,
          typename WrapperClass,
          typename ArgPos>
class BindWrappedArg;

template<typename Invocation, int N>
class BindWrappedArg<Invocation,
                     dax::cont::sig::KeyGroup,
                     dax::cont::sig::Arg<N> >
{
public:
  typedef BindKeyGroup<Invocation,N> type;
};

template<typename Invocation, int N>
class BindWrappedArg<Invocation,
                    dax::cont::arg::Topology::Vertices,
                    dax::cont::sig::Arg<N> >
{
public:
  typedef BindDirect<Invocation,N> type;
};


//find binding finds the correct binding for a parameter
//the main job is to extract out the control signature position and tag
//from Invocation if it exists
template <typename Invocation,
          typename Parameter>
class FindBinding
{
  //use boost grab the result type and rip it out as wrapper type
  //use boost to grab the first argument
  typedef typename boost::function_types::components<Parameter> ComponentTypes;
  typedef typename boost::mpl::at_c<ComponentTypes,0>::type WrappingClassType;
  typedef typename boost::mpl::at_c<ComponentTypes,1>::type IndexType;
public:
  typedef typename BindWrappedArg<Invocation,WrappingClassType,IndexType>::type type;
};

//bind find the specialization for arg binding based on control side tags
template <typename Invocation, int N>
class FindBinding<Invocation, dax::cont::sig::Arg<N> >
{
  typedef typename Invocation::Worklet::DomainType DomainType;
  typedef typename dax::cont::internal::Bindings<Invocation>::type::template GetType<N>::type ControlBinding;
  typedef typename dax::cont::arg::ConceptMapTraits<ControlBinding>::Concept Concept;
  typedef typename dax::cont::arg::ConceptMapTraits<ControlBinding>::Tags Tags;
public:
  //second argument is the control concept type ( Field, Topo ), and the tags
  typedef typename BindArg<DomainType,Concept(Tags),Invocation,N>::type type;
};

//bind workid in the execution signature to the bindworkId class
template <typename Invocation>
class FindBinding<Invocation, dax::cont::sig::WorkId>
{
public:
  typedef BindWorkId<Invocation> type;
};

//bind VertexId directly to the input array (which should be specially
//constructed by the dispatcher)
template<typename Invocation, int N>
class FindBinding<Invocation, dax::cont::sig::VisitIndexArg<N> >
{
public:
  typedef BindDirect<Invocation,N> type;
};

//bind ReductionCount directly to the input array (which should be specially
//constructed by the dispatcher)
template<typename Invocation, int N>
class FindBinding<Invocation, dax::cont::sig::ReductionCountArg<N> >
{
public:
  typedef BindDirect<Invocation,N> type;
};

//bind ReductionOffset directly to the input array (which should be specially
//constructed by the dispatcher)
template<typename Invocation, int N>
class FindBinding<Invocation, dax::cont::sig::ReductionOffsetArg<N> >
{
public:
  typedef BindDirect<Invocation,N> type;
};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_Bind_h
