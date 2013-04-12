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
#ifndef __dax_exec_arg_GeometryCell_h
#define __dax_exec_arg_GeometryCell_h
#if defined(DAX_DOXYGEN_ONLY)

#else // !defined(DAX_DOXYGEN_ONLY)

#include <dax/Types.h>
#include <dax/CellTag.h>

#include <dax/exec/arg/ArgBase.h>
#include <dax/exec/CellVertices.h>
#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/internal/WorkletBase.h>
#include <dax/exec/InterpolatedCellPoints.h>

#include <boost/mpl/if.hpp>
#include <boost/utility/enable_if.hpp>

namespace dax { namespace exec { namespace arg {

template <typename Tags, typename TopologyType, typename PortalType>
class GeometryCell : public dax::exec::arg::ArgBase< GeometryCell<Tags,TopologyType,PortalType> >
{
public:
  //needed for cell type binding to be public
  typedef typename TopologyType::CellTag CellTag;

  typedef dax::exec::arg::ArgBaseTraits< GeometryCell< Tags, TopologyType, PortalType > > Traits;

  typedef typename Traits::ValueType ValueType;
  typedef typename Traits::ReturnType ReturnType;
  typedef typename Traits::SaveType SaveType;

  typedef typename Traits::HasOutTag HasOutTag;

  DAX_CONT_EXPORT GeometryCell(const TopologyType& t,
                               const PortalType& p):
    Topo(t),
    Portal(p),
    Cell(dax::make_Vector3(0.0f,0.0f,0.0f))
    {
    }

  DAX_EXEC_EXPORT ReturnType GetValueForWriting()
    { return this->Cell; }

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType GetValueForReading(
                                const IndexType& index,
                                const dax::exec::internal::WorkletBase& work)
    {
    dax::exec::CellVertices<CellTag> verts =
                                        this->Topo.GetCellConnections(index);
    //now that we the vertices, lets get the coordinates
    this->Cell = dax::exec::internal::FieldGetMultiple(this->Portal,
                                                      verts.GetAsTuple(), work);
    return this->Cell;
    }

  DAX_EXEC_EXPORT void SaveValue(dax::Id index,
                            const dax::exec::internal::WorkletBase& work) const
    {
    this->SaveValue(index,this->Cell,work);
    }

  DAX_EXEC_EXPORT void SaveValue(dax::Id index,
                            const SaveType& values,
                            const dax::exec::internal::WorkletBase& work) const
    {
    dax::exec::CellVertices<CellTag> verts =
                                      this->Topo.GetCellConnections(index);
    //we have the coordinates now stored in the cell, so we have to copy
    //them into the portal
    dax::exec::internal::FieldSetMultiple(this->Portal,
                                          verts.GetAsTuple(),
                                          values.GetAsTuple(), work);
    }
private:
  TopologyType Topo;
  PortalType Portal;
  ValueType Cell;
};

//the traits for GeometryCell
template <typename Tags, typename TopologyType, typename PortalType>
struct ArgBaseTraits< dax::exec::arg::GeometryCell< Tags, TopologyType, PortalType > >
{
  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasOutTag;

  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::In>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasInTag;

  typedef dax::exec::InterpolatedCellPoints<typename TopologyType::CellTag> ValueType;

  typedef typename boost::mpl::if_<typename HasOutTag::type,
                                   ValueType&,
                                   ValueType const&>::type ReturnType;
  typedef ValueType SaveType;
};


}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_GeometryCell_h
