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
#include <dax/exec/InterpolatedCellPoints.h>
#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/internal/WorkletBase.h>

#include <boost/mpl/if.hpp>
#include <boost/utility/enable_if.hpp>

namespace dax { namespace exec { namespace arg {

template <typename Tags, typename TopologyType, typename PortalType>
class GeometryCell
{
public:
  typedef typename TopologyType::CellTag CellTag;
  typedef dax::exec::InterpolatedCellPoints<CellTag> GeometryCellType;

  //if we are going with Out tag
  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   GeometryCellType&,
                                   GeometryCellType const&>::type ReturnType;

  typedef GeometryCellType SaveType;
  typedef GeometryCellType ValueType;

  DAX_CONT_EXPORT GeometryCell(const TopologyType& t,
                             const PortalType& p): Topo(t), Portal(p),Cell() {}

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType operator()(
      const IndexType& index,
      const dax::exec::internal::WorkletBase& work)
  {
    dax::Tuple<dax::Id, GeometryCellType::NUM_VERTICES> verts;
    dax::exec::internal::FieldGetMultiple(this->Topo.CellConnections,
                                dax::CellTraits<CellTag>::NUM_VERTICES * index,
                                verts,
                                work);

    //we have the coordinates now stored in the cell, so we have to copy
    //them into the portal
    this->Cell =  dax::exec::internal::FieldGetMultiple(this->Portal,
                                                        verts, work);
    return this->Cell;
  }

  DAX_EXEC_EXPORT void SaveExecutionResult(int index,
                       const dax::exec::internal::WorkletBase& work) const
    {
    //Look at the concept map traits. If we have the Out tag
    //we know that we must call our TopoExecArgs SaveExecutionResult.
    //Otherwise we are an input argument and that behavior is undefined
    //and very bad things could happen
    typedef typename Tags::
        template Has<typename dax::cont::sig::Out>::type HasOutTag;
    this->saveResult(index,work,HasOutTag());
    }

  //method enabled when we do have the out tag ( or InOut)
  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(dax::Id index,
                  dax::exec::internal::WorkletBase work,
                  HasOutTag,
                  typename boost::enable_if<HasOutTag>::type* = 0) const
    {
    dax::Tuple<dax::Id, GeometryCellType::NUM_VERTICES > verts;
    dax::exec::internal::FieldGetMultiple(this->Topo.CellConnections,
                                dax::CellTraits<CellTag>::NUM_VERTICES * index,
                                verts,
                                work);

    //we have the coordinates now stored in the cell, so we have to copy
    //them into the portal
    dax::exec::internal::FieldSetMultiple(this->Portal, verts,
                                          this->Cell.GetAsTuple(), work);
    }

  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(dax::Id,
                  dax::exec::internal::WorkletBase,
                  HasOutTag,
                  typename boost::disable_if<HasOutTag>::type* = 0) const
    {
    }
private:
  TopologyType Topo;
  PortalType Portal;
  GeometryCellType Cell;
};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_GeometryCell_h
