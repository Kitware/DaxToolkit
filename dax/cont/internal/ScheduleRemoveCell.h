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

#ifndef __dax_cont_internal_ScheduleRemoveCell_h
#define __dax_cont_internal_ScheduleRemoveCell_h

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
#include <dax/cont/internal/ScheduleLowerBoundsAdapter.h>

namespace dax {
namespace exec {
namespace kernel {
namespace internal {

template<class CellType>
struct GetUsedPointsParameters
{
  typename CellType::TopologyType grid;
  dax::exec::Field<dax::Id> outField;
};

template<class CellType>
struct GetUsedPointsFunctor {
  DAX_EXEC_EXPORT void operator()(
      GetUsedPointsParameters<CellType> &parameters,
      dax::Id,dax::Id value,
      const dax::exec::internal::ErrorHandler &errorHandler)
  {
    typedef typename CellType::PointIds PointIds;
    CellType cell(parameters.grid,value);
    PointIds indices = cell.GetPointIndices();
    int* output = parameters.outField.GetArray().GetPointer();
    for(dax::Id i=0;i<CellType::NUM_POINTS;++i)
      {
      output[indices[i]]=1;
      }
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


}
}
}
} //dax::exec::kernel::internal


namespace dax {
namespace cont {
namespace internal {


/// ScheduleRemoveCell is the control enviorment representation of a worklet
//   / of the type WorkDetermineNewCellCount. This class handles properly calling the worklet
/// that the user has defined has being of type WorkDetermineNewCellCount.
///
/// Since ScheduleRemoveCell uses CRTP, every worklet needs to construct a class
/// that inherits from this class and define GenerateParameters.
///
template<class Derived,
         class FunctorClassify,
         class FunctorTopology,
         class DeviceAdapter
         >
class ScheduleRemoveCell
{
public:
  typedef dax::Id MaskType;

  void SetCompactTopology(bool compact)
    {
    this->CompactTopology=compact;
    }

  const bool& GetCompactTopology()
    {
    return this->CompactTopology;
    }

  /// Executes the ScheduleRemoveCell algorithm on the inputGrid and places
  /// the resulting unstructured grid in outGrid
  template<typename InGridType, typename OutGridType>
  void run(const InGridType& inGrid,
           OutGridType& outGrid)
    {
    this->ScheduleClassification(inGrid);
    this->ScheduleTopology(inGrid,outGrid);
    this->GenerateNewTopology(inGrid,outGrid);
    }

/// \fn template <typename WorkType> Parameters GenerateClassificationParameters(const GridType& grid)
/// \brief Abstract method that inherited classes must implement.
///
/// The method must return the populated parameters struct with all the information
/// needed for the ScheduleRemoveCell class to execute the classification \c Functor.

/// \fn template <typename WorkType> Parameters GenerateOutputFields()
/// \brief Abstract method that inherited classes must implement.
///
/// The method is called after the new grids points and topology have been generated
/// This allows dervied classes the ability to use the MaskCellHandle and MaskPointHandle
/// To generate a new field arrays for the output grid

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

    //generate the lower index for each cell id by using inclusive scan,
    //this is required for the lower bounds schedule to generate the key
    //value pair for each id mapping
    dax::cont::ArrayHandle<dax::Id> scanResult;
    dax::Id newTopoSize =
        DeviceAdapter::InclusiveScan(this->NewCellCountHandle,scanResult);

    dax::cont::ArrayHandle<dax::Id> newTopology(newTopoSize);
    dax::cont::internal::ExecutionPackageFieldOutput<dax::Id,DeviceAdapter> packagedTopo(
          newTopology,newTopoSize);

    //call the Topology Functor for each cell in the range that InclusiveScan
    //returned. This will allow us to generate the new topology

    TopoParams params = { packagedGrid.GetExecutionObject(),
                          packagedTopo.GetExecutionObject()};
    dax::cont::internal::ScheduleLowerBounds(FunctorTopology(),
                                             params,
                                             scanResult);
  }

  template<typename InGridType,typename OutGridType>
  void GenerateNewTopology(const InGridType &inGrid, OutGridType& outGrid)
    {

    //The new topology array has been generated by this point.
    //this needs to fill the MaskPointHandleArray


    //generate the point mask using the new topology that has been generated
    dax::cont::ArrayHandle<dax::Id> usedPointIds;
    DeviceAdapter::StreamCompact(this->MaskPointHandle,usedPointIds);
    usedPointIds.CompleteAsOutput();


//    //extract the point coordinates that we need for the new topology
    dax::cont::internal::ExtractCoordinates<DeviceAdapter, InGridType>
                                  extractedCoords(inGrid,usedPointIds);

//    //set the handles to the geometery
//    outGrid = OutGridType(extractedTopology.GetTopology(),
//                          extractedCoords.GetCoordinates());

//    //now that the topology has been fully thresholded,
//    //lets ask our derived class if they need to threshold anything
//    static_cast<Derived*>(this)->GenerateOutputFields();
    }

  template<typename GridType>
  void GeneratePointMask(const GridType &grid,
                         dax::cont::ArrayHandle<dax::Id>& usedCellIds)
    {
    typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
    typedef typename GridPackageType::ExecutionCellType CellType;

    //construct the input grid
    GridPackageType inPGrid(grid);

    const dax::Id size(grid.GetNumberOfPoints());
    this->MaskPointHandle = dax::cont::ArrayHandle<MaskType>(size);

     //we want the size of the points to be based on the numCells * points per cell
    dax::cont::internal::ExecutionPackageFieldOutput<MaskType,DeviceAdapter>
         result(this->MaskPointHandle,size);

    //construct the parameters list for the function
    dax::exec::kernel::internal::GetUsedPointsParameters<CellType> etParams =
                                        {
                                        inPGrid.GetExecutionObject(),
                                        result.GetExecutionObject()
                                        };

    dax::cont::internal::ScheduleMap(
          dax::exec::kernel::internal::GetUsedPointsFunctor<CellType>(),
          etParams,
          usedCellIds);

    this->MaskPointHandle.CompleteAsOutput();

    }

protected:
  bool CompactTopology;
  typedef dax::cont::internal::ExecutionPackageFieldCellOutput<
                                dax::Id,DeviceAdapter> ExecPackFieldCellOutput;
  typedef  boost::shared_ptr< ExecPackFieldCellOutput >
            ExecPackFieldCellOutputPtr;

  ExecPackFieldCellOutputPtr PackageCellCount;

  dax::cont::ArrayHandle<dax::Id> NewCellCountHandle;
  dax::cont::ArrayHandle<MaskType> MaskPointHandle;
};



} //internal
} //exec
} //dax


#endif // __dax_cont_internal_ScheduleRemoveCell_h
