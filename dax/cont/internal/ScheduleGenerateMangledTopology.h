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
namespace exec {
namespace internal {
namespace kernel {

template<class MaskPortalType>
struct ClearUsedPointsFunctor
{
  DAX_CONT_EXPORT
  ClearUsedPointsFunctor(const MaskPortalType &outMask)
    : OutMask(outMask) {  }

  DAX_EXEC_EXPORT void operator()(
      dax::Id index,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    typedef typename MaskPortalType::ValueType MaskType;
    dax::exec::internal::FieldSet(this->OutMask,
                                  index,
                                  static_cast<MaskType>(0),
                                  errorMessage);
  }

private:
  MaskPortalType OutMask;
};

template<class MaskPortalType>
struct GetUsedPointsFunctor
{
  DAX_CONT_EXPORT
  GetUsedPointsFunctor(const MaskPortalType &outMask)
    : OutMask(outMask) {  }

  DAX_EXEC_EXPORT void operator()(
      dax::Id daxNotUsed(key),
      dax::Id value,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage) const
  {
    typedef typename MaskPortalType::ValueType MaskType;
    dax::exec::internal::FieldSet(this->OutMask,
                                  value,
                                  static_cast<MaskType>(1),
                                  errorMessage);
  }

private:
  MaskPortalType OutMask;
};


template<class ArrayPortalType>
struct LowerBoundsInputFunctor
{
  DAX_CONT_EXPORT LowerBoundsInputFunctor(const ArrayPortalType &array)
    : Array(array) {  }

  DAX_EXEC_EXPORT void operator()(
      dax::Id index,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage) const
  {
    dax::exec::internal::FieldSet(this->Array,
                                  index,
                                  index + 1,
                                  errorMessage);
  }

private:
  ArrayPortalType Array;
};

}
}
}
} //dax::exec::internal::kernel


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
public:
  typedef dax::Id MaskType;

protected:
  typedef dax::cont::ArrayHandle<
       dax::Id, ArrayContainerControlTagBasic, DeviceAdapterTag> ArrayHandleId;
  typedef dax::cont::ArrayHandle<
       MaskType, ArrayContainerControlTagBasic, DeviceAdapterTag>
       ArrayHandleMask;

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

    //GeneratePointMask uses the topology that schedule topology generates
    ArrayHandleMask pointMask = this->GeneratePointMask(inGrid, outGrid);

    this->GenerateCompactedTopology(inGrid,outGrid,pointMask);

    //now that the topology has been fully thresholded,
    //lets ask our derived class if they need to threshold anything
    static_cast<Derived*>(this)->GenerateOutputFields(pointMask);
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

  /// \brief Abstract method that inherited classes must implement.
  ///
  /// The method is called after the new grids points and topology have been
  /// generated. \c pointMask has an entry for every point in the input grid
  /// and a positive flag for every entry that corresponds to an output point.
  /// It can be used with dax::cont::internal::StreamCompact.
  ///
  void GenerateOutputFields(const ArrayHandleMask &pointMask);

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
//    //do an inclusive scan of the cell count / cell mask to get the number
//    //of cells in the output
//    ArrayHandleId scannedNewCellCounts;
//    const dax::Id numNewCells =
//        dax::cont::internal::InclusiveScan(newCellCount,
//                                           scannedNewCellCounts,
//                                           DeviceAdapterTag());

//    //fill the validCellRange with the values from 1 to size+1, this is used
//    //for the lower bounds to compute the right indices
//    ArrayHandleId validCellRange;
//    dax::cont::internal::Schedule(
//          dax::exec::internal::kernel::LowerBoundsInputFunctor
//              <typename ArrayHandleId::PortalExecution>(
//                validCellRange.PrepareForOutput(numNewCells)),
//          numNewCells,
//          DeviceAdapterTag());

//    //now do the lower bounds of the cell indices so that we figure out
//    //which original topology indexs match the new indices.
//    dax::cont::internal::LowerBounds(scannedNewCellCounts,
//                                     validCellRange,
//                                     DeviceAdapterTag());

//    // We are done with scannedNewCellCounts.
//    scannedNewCellCounts.ReleaseResources();

    //now call the user topology generation worklet
    dax::cont::internal::ScheduleMap(
          static_cast<Derived*>(this)->CreateTopologyFunctor(inGrid,
                                                             outGrid,
                                                             numNewCells),
          validCellRange,
          outputCellIndexArray);
  }

  template<class InGridType, class OutGridType>
  ArrayHandleMask GeneratePointMask(const InGridType &inGrid,
                                    const OutGridType &outGrid)
    {
    typedef typename ArrayHandleMask::PortalExecution MaskPortalType;

    ArrayHandleMask pointMask;

    // Clear out the mask
    dax::cont::internal::Schedule(
          dax::exec::internal::kernel::ClearUsedPointsFunctor<MaskPortalType>(
            pointMask.PrepareForOutput(inGrid.GetNumberOfPoints())),
          inGrid.GetNumberOfPoints(),
          DeviceAdapterTag());

    // Mark every point that is used at least once.
    // This only works when outGrid is an UnstructuredGrid.
    dax::cont::internal::ScheduleMap(
          dax::exec::internal::kernel::GetUsedPointsFunctor<MaskPortalType>(
            pointMask.PrepareForInPlace()),
          outGrid.GetCellConnections());

    return pointMask;
  }

  template<typename InGridType,typename OutGridType>
  void GenerateCompactedTopology(const InGridType &inGrid,
                                 OutGridType& outGrid,
                                 const ArrayHandleMask &pointMask)
    {
    // Here we are assuming OutGridType is an UnstructuredGrid so that we
    // can set point and connectivity information.

    //extract the point coordinates that we need for the new topology
    dax::cont::internal::StreamCompact(inGrid.GetPointCoordinates(),
                                       pointMask,
                                       outGrid.GetPointCoordinates(),
                                       DeviceAdapterTag());

    typedef typename OutGridType::CellConnectionsType CellConnectionsType;
    typedef typename OutGridType::PointCoordinatesType PointCoordinatesType;

    //compact the topology array to reference the extracted
    //coordinates ids
    {
    // Make usedPointIds become a sorted array of used point indices.
    // If entry i in usedPointIndices is j, then point index i in the
    // output corresponds to point index j in the input.
    ArrayHandleId usedPointIndices;
    dax::cont::internal::Copy(outGrid.GetCellConnections(),
                              usedPointIndices,
                              DeviceAdapterTag());
    dax::cont::internal::Sort(usedPointIndices, DeviceAdapterTag());
    dax::cont::internal::Unique(usedPointIndices, DeviceAdapterTag());
    // Modify the connections of outGrid to point to compacted points.
    dax::cont::internal::LowerBounds(usedPointIndices,
                                     outGrid.GetCellConnections(),
                                     DeviceAdapterTag());
    }
    }
};



} //internal
} //exec
} //dax


#endif // __dax_cont_internal_ScheduleGenerateMangledTopology_h
