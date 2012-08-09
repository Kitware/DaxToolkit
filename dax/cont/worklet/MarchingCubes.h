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

#include <dax/cont/internal/ScheduleGenerateMangledTopology.h>

#include <dax/exec/internal/FieldAccess.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

// ------------------------------------------------------------------- Classify
template<class InputTopologyType,
         class ValuesPortalType,
         class CountPortalType>
struct MarchingCubesClassifyFunctor
{
  // ................................................................ typesdefs
  typedef typename ValuesPortalType::ValueType ValueType;
  typedef typename InputTopologyType::CellType CellType;

  // ..................................................................... cnstr
  DAX_CONT_EXPORT
  MarchingCubesClassifyFunctor(
      const dax::worklet::MarchingCubesClassify<ValueType> &worklet,
      const InputTopologyType                              &inputTopology,
      const ValuesPortalType                               &values,
      const CountPortalType                                &newCellCount)
    : Worklet(worklet),
      InputTopology(inputTopology),
      Values(values),
      NewCellCount(newCellCount) {  }

  // ....................................................................... ()
  DAX_EXEC_EXPORT
  void operator()(
      dax::Id                                        cellIndex,
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
  // ..................................................................... vars
  dax::worklet::MarchingCubesClassify<ValueType> Worklet;
  InputTopologyType                              InputTopology;
  ValuesPortalType                               Values;
  CountPortalType                                NewCellCount;
};

// ------------------------------------------------------------------- Generate
template<class InputTopologyType,
         class ValuesPortalType,
         class CountPortalType,
         class OutputPortalType,
         class OutputTopologyType>
struct MarchingCubesGenerateTopologyFunctor

{
  // ................................................................ typesdefs
  typedef typename InputTopologyType::CellType  InputCellType;
  typedef typename ValuesPortalType::ValueType  ValueType;
  typedef typename CountPortalType::ValueType   CountType;
  typedef typename OutputPortalType::ValueType  OutputType;
  typedef typename OutputTopologyType::CellType OutputCellType;

  // ..................................................................... cnstr
  DAX_CONT_EXPORT
  MarchingCubesGenerateTopologyFunctor(
      const dax::worklet::MarchingCubesTopology<ValueType> &worklet,
      const InputTopologyType                              &inputTopology,
      const ValuesPortalType                               &values,
      const CountPortalType                                &count,
      const OutputPortalType                               &output,
      const OutputTopologyType                             &outputTopology)
    : Worklet(worklet),
      InputTopology(inputTopology),
      Values(values),
      Count(count),
      Output(output),
      OutputTopology(outputTopology) {  }

  // ....................................................................... ()
  DAX_EXEC_EXPORT
  void operator()(
      dax::Id                                        inputCellIndex,
      dax::Id                                        outputCellIndex,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
    const dax::worklet::MarchingCubesTopology<ValueType> &constWorklet = this->Worklet;

    InputCellType inputCell(this->InputTopology, inputCellIndex);
    dax::Tuple<dax::Vector3,3*5> outputPoints;

    constWorklet(inputCell,
                 dax::exec::internal::FieldGetPointsForCell(this->Values,
                                                            inputCell,
                                                            constWorklet),
                 outputPoints);

    for (int localIndex = 0;
         localIndex < this->Count.Get(inputCellIndex)*OutputCellType::NUM_POINTS;
         localIndex++)
      {
      // This only actually works if OutputTopologyType is TopologyUnstructured.
      for(int i =0 ;i<3;++i)
        {
        // This only actually works if OutputTopologyType is TopologyUnstructured.
        dax::exec::internal::FieldSet(this->Output,
                                      outputCellIndex*OutputCellType::NUM_POINTS+localIndex +i,
                                      outputPoints[localIndex][i],
                                      constWorklet);
        }
      dax::exec::internal::FieldSet(this->OutputTopology.CellConnections,
                                   OutputCellType::NUM_POINTS*outputCellIndex+localIndex,
                                   OutputCellType::NUM_POINTS*outputCellIndex+localIndex,
                                   constWorklet);
      }
  }

private:
  // ..................................................................... vars
  dax::worklet::MarchingCubesTopology<ValueType> Worklet;
  InputTopologyType                                               InputTopology;
  ValuesPortalType                                                Values;
  CountPortalType                                                 Count;
  OutputPortalType                                                Output;
  OutputTopologyType                                              OutputTopology;
};

// -------------------------------------------------------------- MarchingCubes
template<class ValueType,
         class Container1,
         class Container2,
         class Adapter>
class MarchingCubes : public dax::cont::internal::ScheduleGenerateMangledTopology
    <
    MarchingCubes<ValueType,Container1,Container2,Adapter>,
    Adapter>
{
protected:
  // ................................................................ typesdefs
  typedef dax::cont::internal::ScheduleGenerateMangledTopology<
      MarchingCubes<ValueType,Container1,Container2,Adapter>, Adapter> Superclass;
  typedef typename Superclass::ArrayHandleId ArrayHandleId;

public:
  // .................................................................... cnstr
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

  // .............................................. CreateClassificationFunctor
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

  // .................................................... CreateTopologyFunctor
  //generate the functor for the topology generation worklet
  template<class InputGridType, class OutputGridType>
  MarchingCubesGenerateTopologyFunctor
  <
    typename InputGridType::TopologyStructConstExecution,
    typename dax::cont::ArrayHandle<ValueType,
                                    Container1,
                                    Adapter>::PortalConstExecution,
    typename ArrayHandleId::PortalConstExecution,
    typename dax::cont::ArrayHandle<ValueType,
                                    Container2,
                                    Adapter>::PortalExecution,
    typename OutputGridType::TopologyStructExecution
  >
  CreateTopologyFunctor(const InputGridType &inputGrid,
                        OutputGridType &outputGrid,
                        dax::Id outputGridSize,
                        const ArrayHandleId &reducedCellCount)
  {
    typedef MarchingCubesGenerateTopologyFunctor<
        typename InputGridType::TopologyStructConstExecution,
        typename dax::cont::ArrayHandle<ValueType,
                                        Container1,
                                        Adapter>::PortalConstExecution,
        typename ArrayHandleId::PortalConstExecution,
        typename dax::cont::ArrayHandle<ValueType,
                                        Container2,
                                        Adapter>::PortalExecution,
        typename OutputGridType::TopologyStructExecution>
      FunctorType;

    typedef typename dax::cont::ArrayHandle<ValueType,
                                            Container2,
                                            Adapter>::PortalExecution
      OutputPortal;

    dax::worklet::MarchingCubesTopology<ValueType> worklet(this->IsoValue);

    FunctorType functor(
          worklet,
          inputGrid.PrepareForInput(),
          this->InputHandle.PrepareForInput(),
          reducedCellCount.PrepareForInput(),
          this->OutputHandle.PrepareForOutput(outputGridSize*3*3),
          outputGrid.PrepareForOutput(outputGridSize*3));

    return functor;
  }

private:
  // ..................................................................... vars
  ValueType                                                         IsoValue;
  const dax::cont::ArrayHandle<ValueType,Container1,Adapter> &InputHandle;
  dax::cont::ArrayHandle<ValueType,Container2,Adapter>      &OutputHandle;
};

}
}
}
} //namespace dax::exec::internal::kernel


namespace dax {
namespace cont {
namespace worklet {

// ------------------------------------------------------------ MarchingCubes()
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
