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

#ifndef __dax_cuda_cont_worklet_CountPointUsage_h
#define __dax_cuda_cont_worklet_CountPointUsage_h

// TODO: This should be auto-generated.

#include <Worklets/MarchingCubes.worklet>

#include <boost/shared_ptr.hpp>

#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

#include <dax/cont/internal/ScheduleGenerateTopology.h>

#include <dax/exec/internal/FieldAccess.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class InputTopologyType, class ValuesPortalType, class CountPortalType>
struct MarchingCubesClassifyFunctor
{
  typedef typename ValuesPortalType::ValueType ValueType;
  typedef typename InputTopologyType::CellType CellType;

  DAX_CONT_EXPORT
  MarchingCubesClassifyFunctor(
      const dax::worklet::MarchingCubesClassify<ValueType> &worklet,
      const InputTopologyType &inputTopology,
      const ValuesPortalType &values,
      const CountPortalType &newCellCount)
    : Worklet(worklet),
      InputTopology(inputTopology),
      Values(values),
      NewCellCount(newCellCount) {  }

  DAX_EXEC_EXPORT void operator()(
      dax::Id cellIndex,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
    const dax::worklet::MarchingCubesClassify<ValueType> &
        constWorklet = this->Worklet;

    CellType cell(this->InputTopology, cellIndex);
    dax::Id newCellCount;

    constWorklet(cell,
                 dax::exec::internal::FieldGetPointsForCell(this->Values,
                                                            cell,
                                                            constWorklet),
                 newCellCount);

    dax::exec::internal::FieldSet(this->NewCellCount,
                                  cellIndex,
                                  newCellCount,
                                  constWorklet);
  }

private:
  dax::worklet::MarchingCubesClassify<ValueType> Worklet;
  InputTopologyType InputTopology;
  ValuesPortalType Values;
  CountPortalType NewCellCount;
};

template<class InputTopologyType, class OutputTopologyType>
struct MarchingCubesGenerateTopologyFunctor
{
  typedef typename InputTopologyType::CellType InputCellType;
  typedef typename OutputTopologyType::CellType OutputCellType;

  DAX_CONT_EXPORT MarchingCubesGenerateTopologyFunctor(
      const dax::worklet::MarchingCubesTopology &worklet,
      const InputTopologyType &inputTopology,
      const OutputTopologyType &outputTopology)
    : Worklet(worklet),
      InputTopology(inputTopology),
      OutputTopology(outputTopology) {  }

  DAX_EXEC_EXPORT void operator()(
      dax::Id outputCellIndex,
      dax::Id inputCellIndex,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
    const dax::worklet::MarchingCubesTopology &constWorklet = this->Worklet;

    InputCellType inputCell(this->InputTopology, inputCellIndex);
    typename OutputCellType::PointConnectionsType outputCellConnections;

    constWorklet(inputCell, outputCellConnections);

    // Write cell connections back to cell array.
    dax::Id index = outputCellIndex * OutputCellType::NUM_POINTS;
    for (int localIndex = 0;
         localIndex < OutputCellType::NUM_POINTS;
         localIndex++)
      {
      // This only actually works if OutputTopologyType is TopologyUnstructured.
      dax::exec::internal::FieldSet(this->OutputTopology.CellConnections,
                                    index,
                                    outputCellConnections[localIndex],
                                    constWorklet);
      index++;
      }
  }

private:
  dax::worklet::MarchingCubesTopology Worklet;
  InputTopologyType InputTopology;
  OutputTopologyType OutputTopology;
};

template<class ValueType,
         class Container1,
         class Container2,
         class Adapter>
class MarchingCubes : public dax::cont::internal::ScheduleGenerateTopology
    <
    MarchingCubes<ValueType,Container1,Container2,Adapter>,
    Adapter>
{
protected:
  typedef dax::cont::internal::ScheduleGenerateTopology<
      MarchingCubes<ValueType,Container1,Container2,Adapter>, Adapter> Superclass;
  typedef typename Superclass::ArrayHandleId ArrayHandleId;
  typedef typename Superclass::ArrayHandleMask ArrayHandleMask;

public:
  //constructor that is passed all the user decided parts of the worklet too
  MarchingCubes(
     const ValueType& isoValue,
     const dax::cont::ArrayHandle<ValueType,Container1,Adapter> &MarchingCubesField,
     dax::cont::ArrayHandle<ValueType,Container2,Adapter> &outputField)
    : IsoValue(isoValue),
      InputHandle(MarchingCubesField),
      OutputHandle(outputField)
  {

  }

  //generate the functor for the classification worklet
  template <class InputGridType>
  MarchingCubesClassifyFunctor<
      typename InputGridType::TopologyStructConstExecution,
      typename dax::cont::ArrayHandle<ValueType,Container1,Adapter>::PortalConstExecution,
      typename ArrayHandleId::PortalExecution>
  CreateClassificationFunctor(const InputGridType &grid,
                              ArrayHandleId &cellCountOutput)
  {
    typedef MarchingCubesClassifyFunctor<
        typename InputGridType::TopologyStructConstExecution,
        typename dax::cont::ArrayHandle<ValueType,Container1,Adapter>::PortalConstExecution,
        typename ArrayHandleId::PortalExecution> FunctorType;

    dax::worklet::MarchingCubesClassify<ValueType> worklet(this->IsoValue);

    FunctorType functor(
          worklet,
          grid.PrepareForInput(),
          this->InputHandle.PrepareForInput(),
          cellCountOutput.PrepareForOutput(grid.GetNumberOfCells()));

    return functor;
  }

  //generate the functor for the topology generation worklet
  template<class InputGridType, class OutputGridType>
  MarchingCubesGenerateTopologyFunctor<
      typename InputGridType::TopologyStructConstExecution,
      typename OutputGridType::TopologyStructExecution>
  CreateTopologyFunctor(const InputGridType &inputGrid,
                        OutputGridType &outputGrid,
                        dax::Id outputGridSize)
  {
    typedef MarchingCubesGenerateTopologyFunctor<
        typename InputGridType::TopologyStructConstExecution,
        typename OutputGridType::TopologyStructExecution> FunctorType;

    dax::worklet::MarchingCubesTopology worklet;

    FunctorType functor(
          worklet,
          inputGrid.PrepareForInput(),
          outputGrid.PrepareForOutput(outputGridSize));

    return functor;
  }

  //MarchingCubes any fields that are needed
  void GenerateOutputFields(const ArrayHandleMask &pointMask)
  {
    //we know that the MarchingCubes is being done on a point field
    dax::cont::internal::StreamCompact(this->InputHandle,
                                       pointMask,
                                       this->OutputHandle,
                                       Adapter());
  }

private:
  ValueType IsoValue;

  const dax::cont::ArrayHandle<ValueType,Container1,Adapter> &InputHandle;
  dax::cont::ArrayHandle<ValueType,Container2,Adapter> &OutputHandle;
};

}
}
}
} //namespace dax::exec::internal::kernel


namespace dax {
namespace cont {
namespace worklet {

template<class InGridType,
         class OutGridType,
         typename ValueType,
         class Container1,
         class Container2,
         class Adapter>
inline void MarchingCubes(
    const InGridType &inGrid,
    OutGridType &outGeom,
    ValueType IsoValue,
    const dax::cont::ArrayHandle<ValueType,Container1,Adapter> &MarchingCubesHandle,
    dax::cont::ArrayHandle<ValueType,Container2,Adapter> &MarchingCubesResult)
{
  dax::exec::internal::kernel::MarchingCubes<
      ValueType,Container1,Container2,Adapter> marchingCubes(IsoValue,
                                                         MarchingCubesHandle,
                                                         MarchingCubesResult);

  marchingCubes.Run(inGrid, outGeom);
}

}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_CountPointUsage_h
