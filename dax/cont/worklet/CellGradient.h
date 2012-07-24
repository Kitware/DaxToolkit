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
#ifndef __dax_cont_worklet_CellGradient_h
#define __dax_cont_worklet_CellGradient_h

// TODO: This should be auto-generated.

#include <Worklets/CellGradient.worklet>

#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

#include <dax/exec/internal/FieldAccess.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class TopologyType,
         class CoordsPortalType,
         class PointFieldPortalType,
         class GradientPortalType>
struct CellGradient {
  typedef typename TopologyType::CellType CellType;

  DAX_CONT_EXPORT
  CellGradient(const dax::worklet::CellGradient &worklet,
               const TopologyType &topology,
               const CoordsPortalType &coords,
               const PointFieldPortalType &pointField,
               const GradientPortalType &gradient)
    : Worklet(worklet),
      Topology(topology),
      Coords(coords),
      PointField(pointField),
      Gradient(gradient) {  }

  DAX_EXEC_EXPORT void operator()(
      dax::Id cellIndex,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
    const dax::worklet::CellGradient &constWorklet = this->Worklet;

    CellType cell(this->Topology, cellIndex);
    typename GradientPortalType::ValueType gradientValue;

    constWorklet(cell,
                 dax::exec::internal::FieldGetPointsForCell(this->Coords,
                                                            cell,
                                                            constWorklet),
                 dax::exec::internal::FieldGetPointsForCell(this->PointField,
                                                            cell,
                                                            constWorklet),
                 gradientValue);

    dax::exec::internal::FieldSet(this->Gradient,
                                  cellIndex,
                                  gradientValue,
                                  constWorklet);
  }

private:
  dax::worklet::CellGradient Worklet;
  const TopologyType &Topology;
  const CoordsPortalType &Coords;
  const PointFieldPortalType &PointField;
  const GradientPortalType &Gradient;
};

}
}
}
} // dax::exec::internal::kernel

namespace dax {
namespace cont {
namespace worklet {

template<class GridType,
         class Container1,
         class Container2,
         class Container3,
         class DeviceAdapter>
DAX_CONT_EXPORT void CellGradient(
    const GridType &grid,
    const dax::cont::ArrayHandle<dax::Vector3,Container1,DeviceAdapter> &coords,
    const dax::cont::ArrayHandle<dax::Scalar,Container2,DeviceAdapter>
        &pointField,
    dax::cont::ArrayHandle<dax::Vector3,Container3,DeviceAdapter> &gradient)
{
  if (coords.GetNumberOfValues() != grid.GetNumberOfPoints())
    {
    throw dax::cont::ErrorControlBadValue(
          "coords size should be same as number of points in grid");
    }
  if (pointField.GetNumberOfValues() != grid.GetNumberOfPoints())
    {
    throw dax::cont::ErrorControlBadValue(
          "pointField size should be same as number of points in grid");
    }

  dax::exec::internal::kernel::CellGradient<
      typename GridType::ExecutionTopologyStruct,
      typename dax::cont::ArrayHandle<dax::Vector3,Container1,DeviceAdapter>::PortalConstExecution,
      typename dax::cont::ArrayHandle<dax::Scalar,Container2,DeviceAdapter>::PortalConstExecution,
      typename dax::cont::ArrayHandle<dax::Vector3,Container3,DeviceAdapter>::PortalExecution>
      kernel(dax::worklet::CellGradient(),
             grid.PrepareForInput(),
             coords.PrepareForInput(),
             pointField.PrepareForInput(),
             gradient.PrepareForOutput(grid.GetNumberOfCells()));

  dax::cont::internal::Schedule(kernel,
                                grid.GetNumberOfCells(),
                                DeviceAdapter());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_CellGradient_h
