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

#ifndef __dax_cont_internal_ScheduleGenerateMangledTopology_h
#define __dax_cont_internal_ScheduleGenerateMangledTopology_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/WorkletGenerateTopology.h>

#include <dax/exec/internal/ErrorMessageBuffer.h>
#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/internal/GridTopologies.h>

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ScheduleMapAdapter.h>

namespace dax {
namespace cont {
namespace internal {

/// ScheduleGenerateMangledTopology is the control enviorment representation of a
/// worklet of the type WorkDetermineNewCellCount. This class handles properly
/// calling the worklet that the user has defined has being of type
/// WorkDetermineNewCellCount.
///
/// Since ScheduleGenerateMangledTopology uses CRTP, every worklet needs to construct
/// a class that inherits from this class and define GenerateParameters.
///
template<class Derived, class DeviceAdapterTag>
class ScheduleGenerateMangledTopology
{
protected:
  typedef dax::cont::ArrayHandle<
       dax::Id, ArrayContainerControlTagBasic, DeviceAdapterTag> ArrayHandleId;

public:
  /// Executes the ScheduleGenerateMangledTopology algorithm on the inputGrid and
  /// places the resulting unstructured grid in outGrid
  ///
  template<typename InGridType, typename OutGridType>
  DAX_CONT_EXPORT
  void Run(const InGridType& inGrid,
           OutGridType& outGrid)
    {
    ArrayHandleId cellCount = this->ScheduleClassification(inGrid);

    // Shrink();
    ArrayHandleId validCellRange;
    cellCount.PrepareForInput();
    dax::cont::internal::StreamCompact(cellCount,
                                       validCellRange,
                                       DeviceAdapterTag());
    ArrayHandleId reducedCellCount;
    cellCount.PrepareForInput();
    dax::cont::internal::StreamCompact(cellCount,
                                       cellCount,
                                       reducedCellCount,
                                       DeviceAdapterTag());
    // CalculateTotalOuputCells() + MakeOuputCellIndexArray();
    ArrayHandleId outputCellIndexArray;
    reducedCellCount.PrepareForInput();
    //outputCellIndexArray.PrepareForOutput();
    dax::Id totalOutputCells = dax::cont::internal::ExclusiveScan(reducedCellCount,
                                                                  outputCellIndexArray,
                                                                  DeviceAdapterTag());

    // Generate();
    this->ScheduleTopology(inGrid, outGrid,
                           validCellRange,
                           reducedCellCount,
                           outputCellIndexArray,
                           totalOutputCells);
    newCellCount.ReleaseResources();
    }

#ifdef DAX_DOXYGEN_ONLY

  /// \brief Abstract method that inherited classes must implement.
  ///
  /// This method must return a fully constructed functor object ready to be
  /// passed to dax::cont::internal::Schedule. The functor should populate
  /// cellCountOutput with the number of cells to be constructed in the output
  /// for each cell in the input.
  ///
  template<class GridType>
  FunctorClassify CreateClassificationFunctor(const GridType& grid,
                                              ArrayHandleId &cellCountOutput);

  /// \brief Abstract method that inherited classes must implement.
  ///
  /// This method must return a fully constructed functor object ready to be
  /// passed to dax::cont::internal::ScheduleMap. The functor should populate
  /// the connections array in outputGrid. In the schedule map the "key" is the
  /// output cell index and the "value" is the input cell index.
  ///
  template<class InputGridType, class OutputGridType>
  FunctorTopology CreateTopologyFunctor(const InputGridType &inputGrid,
                                        OutputGridType &outputGrid,
                                        dax::Id outputGridSize);


#endif //DAX_DOXYGEN_ONLY

private:

  //constructs everything needed to call the user defined worklet
  template<typename InGridType>
  ArrayHandleId ScheduleClassification(const InGridType &grid)
  {
    ArrayHandleId newCellCount;

    //we need the control grid to create the parameters struct.
    //So pass those objects to the derived class and let it populate the
    //functor struct with all the user defined information letting the user
    //defined class do this work allows us to easily extend this class
    //for an arbitrary number of input parameters
    dax::cont::internal::Schedule(
          static_cast<Derived*>(this)->CreateClassificationFunctor(
            grid, newCellCount),
          grid.GetNumberOfCells(),
          DeviceAdapterTag());

    return newCellCount;
  }

  template<typename InGridType, typename OutGridType>
  void ScheduleTopology(const InGridType& inGrid,
                        OutGridType& outGrid,
                        const ArrayHandleId validCellRange,
                        const ArrayHandleId reducedCellCount,
                        const ArrayHandleId outputCellIndexArray,
                        const dax::Id totalOutputCells)
  {
          static_cast<Derived*>(this)->CreateTopologyFunctor(inGrid,
                                                             outGrid,
                                                             numNewCells),
          validCellRange,
          outputCellIndexArray);
  }
};
} //internal
} //exec
} //dax


#endif // __dax_cont_internal_ScheduleGenerateMangledTopology_h
