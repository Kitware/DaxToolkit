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
#ifndef __dax_exec_arg_TopologyCell_h
#define __dax_exec_arg_TopologyCell_h

#include <dax/Types.h>

#include <dax/exec/arg/ArgBase.h>
#include <dax/exec/CellVertices.h>
#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/internal/WorkletBase.h>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>

namespace dax { namespace exec { namespace arg {

template <typename Tags, typename TopologyType>
class TopologyCell : public dax::exec::arg::ArgBase< TopologyCell<Tags, TopologyType> >
{
public:
  //needed for cell type binding to be public
  typedef typename TopologyType::CellTag CellTag;

  typedef dax::exec::arg::ArgBaseTraits< TopologyCell< Tags, TopologyType > > Traits;

  typedef typename Traits::ValueType ValueType;
  typedef typename Traits::ReturnType ReturnType;
  typedef typename Traits::SaveType SaveType;

  DAX_CONT_EXPORT TopologyCell(const TopologyType& t):
    Topo(t),
    Cell(0)
    {
    }

  DAX_EXEC_EXPORT ReturnType GetValueForWriting()
    { return this->Cell; }

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType GetValueForReading(
                            const IndexType& index,
                            const dax::exec::internal::WorkletBase& work) const
    {
    (void)work;  // Shut up compiler.
    DAX_ASSERT_EXEC(index >= 0, work);
    DAX_ASSERT_EXEC(index < Topo.GetNumberOfCells(), work);
    return this->Topo.GetCellConnections(index);
    }

  DAX_EXEC_EXPORT void SaveValue(int index,
                       const dax::exec::internal::WorkletBase& work) const
    { this->SaveValue(index,this->Cell,work); }

  DAX_EXEC_EXPORT void SaveValue(int index, const SaveType& values,
                       const dax::exec::internal::WorkletBase& work) const
    {
    dax::exec::internal::FieldSetMultiple(this->Topo.CellConnections,
                                dax::CellTraits<CellTag>::NUM_VERTICES * index,
                                values.GetAsTuple(),
                                work);
    }

private:
  TopologyType Topo;
  ValueType Cell;
};

//the traits for TopologyCell
template <typename Tags, typename TopologyType>
struct ArgBaseTraits< dax::exec::arg::TopologyCell< Tags, TopologyType > >
{
  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasOutTag;

  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::In>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasInTag;

  typedef dax::exec::CellVertices<typename TopologyType::CellTag> ValueType;
  typedef typename boost::mpl::if_<typename HasOutTag::type,
                                   ValueType&,
                                   ValueType const>::type ReturnType;
  typedef ValueType SaveType;
};

} } } //namespace dax::exec::arg



#endif //__dax_exec_arg_TopologyCell_h
