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

#include <boost/shared_ptr.hpp>

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/WorkGenerateTopology.h>

#include <dax/internal/GridTopologys.h>

#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>
#include <dax/cont/internal/ExtractCoordinates.h>
#include <dax/cont/internal/ScheduleMapAdapter.h>

namespace dax {
namespace exec {
namespace kernel {
namespace internal {

template<class CellType>
struct GetUsedPointsParameters
{
  dax::exec::Field<dax::Id> outField;
};

template<class CellType>
struct GetUsedPointsFunctor {
  DAX_EXEC_EXPORT void operator()(
      GetUsedPointsParameters<CellType> &parameters,
      dax::Id /*key*/,dax::Id value,
      const dax::exec::internal::ErrorHandler &errorHandler)
  {
    int* output = parameters.outField.GetArray().GetPointer();
    output[value]=1;
  }
};

template<class ICT, class OCT>
struct GenerateTopologyParameters
{
  typedef ICT CellType;
  typedef OCT OutCellType;
  typedef typename CellType::TopologyType GridType;

  GridType grid;
  dax::exec::Field<dax::Id> outputTopology;
};


struct LowerBoundsInputFunctor
{
DAX_EXEC_EXPORT void operator()(dax::internal::DataArray<dax::Id> array,
                                dax::Id index,
                                dax::exec::internal::ErrorHandler &)
{
  array.SetValue(index, index+1);
}
};

}
}
}
} //dax::exec::kernel::internal


namespace dax {
namespace cont {
namespace internal {


/// ScheduleGenerateTopology is the control enviorment representation of a worklet
//   / of the type WorkDetermineNewCellCount. This class handles properly calling the worklet
/// that the user has defined has being of type WorkDetermineNewCellCount.
///
/// Since ScheduleGenerateTopology uses CRTP, every worklet needs to construct a class
/// that inherits from this class and define GenerateParameters.
///
template<class Derived,
         class FunctorClassify,
         class FunctorTopology,
         class DeviceAdapter
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
  template <typename GridType, typename PackagedGrid>
  ParametersClassify GenerateClassificationParameters(const GridType& grid,
                                                      PackagedGrid& pgrid);

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
    typedef dax::cont::internal::ExecutionPackageGrid<InGridType> GridPackageType;
    //create the grid, and result packages
    GridPackageType packagedGrid(grid);

    this->NewCellCountHandle =
        dax::cont::ArrayHandle<dax::Id>(grid.GetNumberOfCells());

    this->PackageCellCount = ExecPackFieldCellOutputPtr(
                          new ExecPackFieldCellOutput(
                                this->NewCellCountHandle,grid));

    //we need the control grid to create the parameters struct.
    //So pass those objects to the derived class and let it populate the
    //parameters struct with all the user defined information letting the user
    //defined class do this work allows us to easily extend this class
    //for an arbitrary number of input parameters
    //Actually run the FunctorClassify which is the user worklet with the correct parameters
    DeviceAdapter::Schedule(FunctorClassify(),
                            static_cast<Derived*>(this)->GenerateClassificationParameters(grid,packagedGrid),
                            grid.GetNumberOfCells());

    //mark the handle as complete so we can use as input
    this->NewCellCountHandle.CompleteAsOutput();
    }

  template<typename InGridType, typename OutGridType>
  void ScheduleTopology(const InGridType& inGrid, const OutGridType& )
  {
    typedef dax::cont::internal::ExecutionPackageGrid<InGridType> GridPackageType;
    typedef typename GridPackageType::ExecutionCellType InCellType;

    typedef dax::cont::internal::ExecutionPackageGrid<OutGridType> OutGridPackageType;
    typedef typename OutGridPackageType::ExecutionCellType OutCellType;


    typedef dax::exec::kernel::internal::GenerateTopologyParameters<InCellType,OutCellType> TopoParams;
    //create the grid, and result packages
    GridPackageType packagedGrid(inGrid);

    //do an inclusive scan of the cell count / cell mask to get the number
    //of cells in the output
    dax::cont::ArrayHandle<dax::Id> newCellCounts(
          this->NewCellCountHandle.GetNumberOfEntries());
    const dax::Id newNumOfCells =
        DeviceAdapter::InclusiveScan(this->NewCellCountHandle,newCellCounts);
    newCellCounts.CompleteAsOutput();

    //fill the validCellRange with the values from 1 to size+1, this is used
    //for the lower bounds to compute the right indices
    dax::cont::ArrayHandle<dax::Id,DeviceAdapter> validCellRange(newNumOfCells);
    DeviceAdapter::Schedule(dax::exec::kernel::internal::LowerBoundsInputFunctor(),
                            validCellRange.ReadyAsOutput(),
                            newNumOfCells);
    validCellRange.CompleteAsOutput();

    //now do the lower bounds of the cell indices so that we figure out
    //which original topology indexs match the new indices.
    DeviceAdapter::LowerBounds(newCellCounts,validCellRange,validCellRange);
    validCellRange.CompleteAsOutput();

    //we can now remove newCellCounts
    newCellCounts.ReleaseExecutionResources();

    //now call the user topology generation worklet
    //now we can determine the size of the topoogy and construct that array
    const dax::Id genTopoSize(OutCellType::NUM_POINTS * newNumOfCells);
    this->GeneratedTopology = dax::cont::ArrayHandle<dax::Id> (genTopoSize);
    dax::cont::internal::ExecutionPackageFieldOutput<dax::Id,DeviceAdapter> packagedTopo(
          this->GeneratedTopology,genTopoSize);

    TopoParams params = { packagedGrid.GetExecutionObject(),
                          packagedTopo.GetExecutionObject()};


    dax::cont::internal::ScheduleMap(FunctorTopology(),params,validCellRange);
  }

  template<typename GridType>
  void GeneratePointMask(const GridType &grid)
    {
    typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
    typedef typename GridPackageType::ExecutionCellType CellType;

    const dax::Id size(grid.GetNumberOfPoints());
    this->MaskPointHandle = dax::cont::ArrayHandle<MaskType>(size);

     //we want the size of the points to be based on the numCells * points per cell
    dax::cont::internal::ExecutionPackageFieldOutput<MaskType,DeviceAdapter>
         result(this->MaskPointHandle,size);

    //construct the parameters list for the function
    dax::exec::kernel::internal::GetUsedPointsParameters<CellType> etParams =
                                        {
                                        result.GetExecutionObject()
                                        };

    dax::cont::internal::ScheduleMap(
          dax::exec::kernel::internal::GetUsedPointsFunctor<CellType>(),
          etParams,
          this->GeneratedTopology);

    this->MaskPointHandle.CompleteAsOutput();

    }

  template<typename InGridType,typename OutGridType>
  void GenerateCompactedTopology(const InGridType &inGrid, OutGridType& outGrid)
    {
    //generate the point mask using the new topology that has been generated
    dax::cont::ArrayHandle<dax::Id> usedPointIds;
    DeviceAdapter::StreamCompact(this->MaskPointHandle,usedPointIds);
    usedPointIds.CompleteAsOutput();

    //extract the point coordinates that we need for the new topology
    dax::cont::internal::ExtractCoordinates<DeviceAdapter, InGridType>
                                  extractedCoords(inGrid,usedPointIds);

    //compact the topology array to reference the extracted
    //coordinates ids
    {
    dax::cont::ArrayHandle<dax::Id,DeviceAdapter> temp(
          this->GeneratedTopology.GetNumberOfEntries());
    DeviceAdapter::Copy(this->GeneratedTopology,temp);
    DeviceAdapter::Sort(temp);
    DeviceAdapter::Unique(temp);
    DeviceAdapter::LowerBounds(temp,this->GeneratedTopology,
                               this->GeneratedTopology);
    }

    this->GeneratedTopology.CompleteAsOutput();
    //set the handles to the geometery, this actually changes the output
    //geometery
    outGrid = OutGridType(this->GeneratedTopology,
                          extractedCoords.GetCoordinates());

    }

protected:
  bool CompactTopology;
  typedef dax::cont::internal::ExecutionPackageFieldCellOutput<
                                dax::Id,DeviceAdapter> ExecPackFieldCellOutput;
  typedef  boost::shared_ptr< ExecPackFieldCellOutput >
            ExecPackFieldCellOutputPtr;

  ExecPackFieldCellOutputPtr PackageCellCount;

  //Holds the new cell count per old cell, this can be used
  //as the Mask for cell arrays
  dax::cont::ArrayHandle<dax::Id> NewCellCountHandle;

  dax::cont::ArrayHandle<dax::Id> GeneratedTopology;

  //Holds the mask for point Arrays
  dax::cont::ArrayHandle<MaskType> MaskPointHandle;
};



} //internal
} //exec
} //dax


#endif // __dax_cont_internal_ScheduleGenerateTopology_h
