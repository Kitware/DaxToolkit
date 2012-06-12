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

#ifndef __dax_cont_internal_ScheduleGenerateTopology_h
#define __dax_cont_internal_ScheduleGenerateTopology_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkGenerateTopology.h>

#include <dax/exec/internal/ExecutionAdapter.h>
#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/internal/GridTopologies.h>
#include <dax/exec/internal/WorkEmpty.h>

#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>
#include <dax/cont/internal/ExtractCoordinates.h>
#include <dax/cont/internal/ScheduleMapAdapter.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class MaskType, class ExecAdapter>
struct GetUsedPointsParameters
{
  dax::exec::FieldPointOut<MaskType, ExecAdapter> outField;
};

template<class MaskType, class ExecAdapter>
struct ClearUsedPointsFunctor
{
  DAX_EXEC_EXPORT void operator()(
      const GetUsedPointsParameters<MaskType, ExecAdapter> &parameters,
      dax::Id index,
      const ExecAdapter &execAdapter)
  {
    dax::exec::internal::WorkEmpty<ExecAdapter> dummywork(execAdapter);
    dax::exec::internal::FieldAccess::SetField(parameters.outField,
                                               index,
                                               static_cast<MaskType>(0),
                                               dummywork);
  }
};

template<class MaskType, class ExecAdapter>
struct GetUsedPointsFunctor {
  DAX_EXEC_EXPORT void operator()(
      const GetUsedPointsParameters<MaskType, ExecAdapter> &parameters,
      dax::Id /*key*/,
      dax::Id value,
      const ExecAdapter &execAdapter) const
  {
    dax::exec::internal::WorkEmpty<ExecAdapter> dummywork(execAdapter);
    dax::exec::internal::FieldAccess::SetField(parameters.outField,
                                               value,
                                               static_cast<MaskType>(1),
                                               dummywork);
  }
};

template<class ICT, class OCT, class ExecAdapter>
struct GenerateTopologyParameters
{
  typedef ICT CellType;
  typedef OCT OutCellType;
  typedef typename CellType::template GridStructures<ExecAdapter>::TopologyType
      GridType;

  GridType grid;
  dax::exec::FieldOut<dax::Id, ExecAdapter> outputConnections;
};


template<class ExecAdapter>
struct LowerBoundsInputFunctor
{
DAX_EXEC_EXPORT void operator()(
    dax::exec::FieldOut<dax::Id, ExecAdapter> field,
    dax::Id index,
    const ExecAdapter &execAdapter) const
{
  dax::exec::internal::WorkEmpty<ExecAdapter> dummywork(execAdapter);
  dax::exec::internal::FieldAccess::SetField(field, index, index+1, dummywork);
}
};

}
}
}
} //dax::exec::internal::kernel


namespace dax {
namespace cont {
namespace internal {


/// ScheduleGenerateTopology is the control enviorment representation of a
/// worklet of the type WorkDetermineNewCellCount. This class handles properly
/// calling the worklet that the user has defined has being of type
/// WorkDetermineNewCellCount.
///
/// Since ScheduleGenerateTopology uses CRTP, every worklet needs to construct
/// a class that inherits from this class and define GenerateParameters.
///
template<class Derived,
         class FunctorClassify,
         class FunctorTopology,
         class ArrayContainerControlTag,
         class DeviceAdapterTag
         >
class ScheduleGenerateTopology
{
public:
  typedef dax::Id MaskType;

  /// Executes the ScheduleGenerateTopology algorithm on the inputGrid and places
  /// the resulting unstructured grid in outGrid
  template<typename InGridType, typename OutGridType>
  void run(const InGridType& inGrid,
           OutGridType& outGrid)
    {
    this->ScheduleClassification(inGrid);

    this->ScheduleTopology(inGrid,outGrid);

    //GeneratePointMask uses the topology that schedule topology generates
    this->GeneratePointMask(inGrid);

    this->GenerateCompactedTopology(inGrid,outGrid);

    //now that the topology has been fully thresholded,
    //lets ask our derived class if they need to threshold anything
    static_cast<Derived*>(this)->GenerateOutputFields();
    }

#ifdef DAX_DOXYGEN_ONLY

  /// \brief Abstract method that inherited classes must implement.
  ///
  /// The method must return the populated parameters struct with all the
  /// information needed for the ScheduleGenerateTopology class to execute the
  /// classification \c Functor.
  ///
  template <typename GridType, typename ExecutionTopologyType>
  ParametersClassify GenerateClassificationParameters(
      const GridType& grid,
      const ExecutionTopologyType& executionTopology);

  /// \brief Abstract method that inherited classes must implement.
  ///
  /// The method is called after the new grids points and topology have been
  /// generated. This allows dervied classes the ability to use the
  /// MaskCellHandle and MaskPointHandle To generate a new field arrays for the
  /// output grid
  ///
  void GenerateOutputFields();

#endif //DAX_DOXYGEN_ONLY

protected:

  //constructs everything needed to call the user defined worklet
  template<typename InGridType>
  void ScheduleClassification(const InGridType &grid)
  {
    typedef typename InGridType::ExecutionTopologyStruct InExecTopology;
    InExecTopology execTopology =
        dax::cont::internal::ExecutionPackageGrid(grid);

    this->NewCellCountField =
        dax::cont::internal::ExecutionPackageFieldGrid<dax::exec::FieldCellOut>(
          this->NewCellCountHandle, grid);

    //we need the control grid to create the parameters struct.
    //So pass those objects to the derived class and let it populate the
    //parameters struct with all the user defined information letting the user
    //defined class do this work allows us to easily extend this class
    //for an arbitrary number of input parameters
    //Actually run the FunctorClassify which is the user worklet with the
    //correct parameters
    dax::cont::internal::Schedule(
          FunctorClassify(),
          static_cast<Derived*>(this)
          ->GenerateClassificationParameters(grid, execTopology),
          grid.GetNumberOfCells(),
          ArrayContainerControlTag(),
          DeviceAdapterTag());
  }

  template<typename InGridType, typename OutGridType>
  void ScheduleTopology(const InGridType& inGrid, const OutGridType& )
  {
    typedef typename InGridType::ExecutionTopologyStruct InExecTopology;
    typedef typename InExecTopology::CellType InCellType;

    typedef typename OutGridType::ExecutionTopologyStruct OutExecTopology;
    typedef typename OutExecTopology::CellType OutCellType;

    typedef dax::exec::internal::kernel
        ::GenerateTopologyParameters<InCellType,OutCellType,ExecutionAdapter>
        TopologyParams;

    InExecTopology inExecTopology
        = dax::cont::internal::ExecutionPackageGrid(inGrid);

    //do an inclusive scan of the cell count / cell mask to get the number
    //of cells in the output
    dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTag, DeviceAdapterTag>
        scannedNewCellCounts;
    const dax::Id newNumCells =
        dax::cont::internal::InclusiveScan(this->NewCellCountHandle,
                                           scannedNewCellCounts,
                                           DeviceAdapterTag());

    //fill the validCellRange with the values from 1 to size+1, this is used
    //for the lower bounds to compute the right indices
    dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTag, DeviceAdapterTag>
        validCellRange;
    dax::exec::FieldOut<dax::Id, ExecutionAdapter> validCellRangeField
        = dax::cont::internal::ExecutionPackageFieldArray<dax::exec::FieldOut>(
          validCellRange, newNumCells);
    dax::cont::internal::Schedule(
          dax::exec::internal::kernel::LowerBoundsInputFunctor
              <ExecutionAdapter>(),
          validCellRangeField,
          newNumCells,
          ArrayContainerControlTag(),
          DeviceAdapterTag());

    //now do the lower bounds of the cell indices so that we figure out
    //which original topology indexs match the new indices.
    dax::cont::internal::LowerBounds(scannedNewCellCounts,
                                     validCellRange,
                                     DeviceAdapterTag());

    // We are done with scannedNewCellCounts.
    scannedNewCellCounts.ReleaseResources();

    //now call the user topology generation worklet
    //now we can determine the size of the topoogy and construct that array
    const dax::Id generatedConnectionSize = OutCellType::NUM_POINTS*newNumCells;
    dax::exec::FieldOut<dax::Id, ExecutionAdapter> generatedConnectionsField
        = dax::cont::internal::ExecutionPackageFieldArray<dax::exec::FieldOut>(
          this->GeneratedConnectionsHandle,generatedConnectionSize);

    TopologyParams parameters;
    parameters.grid = inExecTopology;
    parameters.outputConnections = generatedConnectionsField;

    dax::cont::internal::ScheduleMap(FunctorTopology(),
                                     parameters,
                                     validCellRange);
  }

  template<typename GridType>
  void GeneratePointMask(const GridType &grid)
    {
    typedef typename GridType::ExecutionTopologyStruct ExecTopology;
    typedef typename ExecTopology::CellType CellType;

    dax::exec::FieldPointOut<dax::Id, ExecutionAdapter> maskPointsField
        = dax::cont::internal::ExecutionPackageFieldGrid<
          dax::exec::FieldPointOut>(this->MaskPointHandle, grid);

    //construct the parameters list for the function
    dax::exec::internal::kernel
        ::GetUsedPointsParameters<MaskType, ExecutionAdapter> parameters;
    parameters.outField = maskPointsField;

    //clear out the mask
    dax::cont::internal::Schedule(
          dax::exec::internal::kernel::ClearUsedPointsFunctor<MaskType,ExecutionAdapter>(),
          parameters,
          grid.GetNumberOfPoints(),
          ArrayContainerControlTag(),
          DeviceAdapterTag());

    dax::cont::internal::ScheduleMap(
          dax::exec::internal::kernel::GetUsedPointsFunctor<MaskType, ExecutionAdapter>(),
          parameters,
          this->GeneratedConnectionsHandle);
  }

  template<typename InGridType,typename OutGridType>
  void GenerateCompactedTopology(const InGridType &inGrid, OutGridType& outGrid)
    {
    //generate the point mask using the new topology that has been generated
    dax::cont::ArrayHandle<dax::Id,ArrayContainerControlTag,DeviceAdapterTag>
        usedPointIds;
    dax::cont::internal::StreamCompact(this->MaskPointHandle,
                                       usedPointIds,
                                       DeviceAdapterTag());

    //extract the point coordinates that we need for the new topology
    dax::cont::ArrayHandle<
        dax::Vector3,ArrayContainerControlTag,DeviceAdapterTag>
        coordinates = dax::cont::internal::ExtractCoordinates(inGrid,
                                                              usedPointIds);

    //compact the topology array to reference the extracted
    //coordinates ids
    {
    dax::cont::ArrayHandle<dax::Id,ArrayContainerControlTag,DeviceAdapterTag>
        temp;
    dax::cont::internal::Copy(this->GeneratedConnectionsHandle,
                              temp,
                              DeviceAdapterTag());
    dax::cont::internal::Sort(temp, DeviceAdapterTag());
    dax::cont::internal::Unique(temp, DeviceAdapterTag());
    dax::cont::internal::LowerBounds(temp,
                                     this->GeneratedConnectionsHandle,
                                     DeviceAdapterTag());
    }

    //set the handles to the geometery, this actually changes the output
    //geometery
    outGrid.SetCellConnections(this->GeneratedConnectionsHandle);
    outGrid.SetPointCoordinates(coordinates);
    }

protected:
  bool CompactTopology;

  typedef dax::exec::internal
      ::ExecutionAdapter<ArrayContainerControlTag, DeviceAdapterTag>
        ExecutionAdapter;

  //Holds the new cell count per old cell, this can be used
  //as the Mask for cell arrays
  dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTag, DeviceAdapterTag>
      NewCellCountHandle;

  dax::exec::FieldCellOut<dax::Id, ExecutionAdapter> NewCellCountField;

  dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTag, DeviceAdapterTag>
      GeneratedConnectionsHandle;

  //Holds the mask for point Arrays
  dax::cont::ArrayHandle<MaskType, ArrayContainerControlTag, DeviceAdapterTag>
      MaskPointHandle;
};



} //internal
} //exec
} //dax


#endif // __dax_cont_internal_ScheduleGenerateTopology_h
