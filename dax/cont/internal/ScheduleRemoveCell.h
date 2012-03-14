#ifndef __dax_exec_internal_ScheduleRemoveCell_h
#define __dax_exec_internal_ScheduleRemoveCell_h

#include <boost/shared_ptr.hpp>

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>

#include <dax/internal/GridTopologys.h>
#include <dax/exec/internal/FieldAccess.h>

#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>
#include <dax/cont/internal/ExtractTopology.h>
#include <dax/cont/internal/ExtractCoordinates.h>
#include <dax/cont/internal/ScheduleMapAdapter.h>

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

}
}
}
} //dax::exec::kernel::internal


namespace dax {
namespace cont {
namespace internal {


/// ScheduleRemoveCell is the control enviorment representation of a worklet
//   / of the type WorkRemoveCell. This class handles properly calling the worklet
/// that the user has defined has being of type WorkRemoveCell.
///
/// Since ScheduleRemoveCell uses CRTP, every worklet needs to construct a class
/// that inherits from this class and define GenerateParameters.
///
template<class Derived,
         class Functor,
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
    this->ScheduleWorklet(inGrid);
    this->GenerateOutput(inGrid,outGrid);
    }

/// \fn template <typename WorkType> Parameters GenerateParameters(const GridType& grid)
/// \brief Abstract method that inherited classes must implement.
///
/// The method must return the populated parameters struct with all the information
/// needed for the ScheduleRemoveCell class to execute the \c Functor.

/// \fn template <typename WorkType> Parameters GenerateOutputFields()
/// \brief Abstract method that inherited classes must implement.
///
/// The method is called after the new grids points and topology have been generated
/// This allows dervied classes the ability to use the MaskCellHandle and MaskPointHandle
/// To generate a new field arrays for the output grid

protected:

  //constructs everything needed to call the user defined worklet
  template<typename InGridType>
  void ScheduleWorklet(const InGridType &grid)
    {
    typedef dax::cont::internal::ExecutionPackageGrid<InGridType> GridPackageType;
    //create the grid, and result packages
    GridPackageType packagedGrid(grid);

    this->MaskCellHandle =
        dax::cont::ArrayHandle<MaskType>(grid.GetNumberOfCells());

    this->PackageMaskCell = ExecPackFieldCellOutputPtr(
                          new ExecPackFieldCellOutput(
                                this->MaskCellHandle,grid));

    //we need the control grid to create the parameters struct.
    //So pass those objects to the derived class and let it populate the
    //parameters struct with all the user defined information letting the user
    //defined class do this work allows us to easily extend this class
    //for an arbitrary number of input parameters
    //Actually run the Functor which is the user worklet with the correct parameters
    DeviceAdapter::Schedule(Functor(),
                            static_cast<Derived*>(this)->GenerateParameters(
                            grid,packagedGrid),
                            grid.GetNumberOfCells());

    //mark the handle as complete so we can use as input
    this->MaskCellHandle.CompleteAsOutput();
    }

  template<typename InGridType,typename OutGridType>
  void GenerateOutput(const InGridType &inGrid, OutGridType& outGrid)
    {

    //stream compact with two paramters the second one needs to be dax::Ids
    dax::cont::ArrayHandle<dax::Id> usedCellIds;
    DeviceAdapter::StreamCompact(this->MaskCellHandle,usedCellIds);    
    usedCellIds.CompleteAsOutput(); //mark it complete so we can use as input

    if(usedCellIds.GetNumberOfEntries() == 0)
      {
      //we have nothing to generate so return the output unmodified
      return;
      }

    if(this->CompactTopology)
      {
      //extract from the grid the subset of topology information we
      //need to construct the unstructured grid
      dax::cont::internal::ExtractTopology<DeviceAdapter, InGridType>
         extractedTopology(inGrid, usedCellIds, this->CompactTopology);

      //generate the point mask
      this->GeneratePointMask(inGrid,usedCellIds);

      //now that the topology has been fully thresholded,
      //lets ask our derived class if they need to threshold anything
      static_cast<Derived*>(this)->GenerateOutputFields();

      dax::cont::ArrayHandle<dax::Id> usedPointIds;
      DeviceAdapter::StreamCompact(this->MaskPointHandle,usedPointIds);
      usedPointIds.CompleteAsOutput();

      //extract the point coordinates that we need
      dax::cont::internal::ExtractCoordinates<DeviceAdapter, InGridType>
                                      extractedCoords(inGrid,usedPointIds);

      //set the handles to the geometery
      outGrid = OutGridType(extractedTopology.GetTopology(),
                            extractedCoords.GetCoordinates());
      }    
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
                                MaskType,DeviceAdapter> ExecPackFieldCellOutput;
  typedef  boost::shared_ptr< ExecPackFieldCellOutput >
            ExecPackFieldCellOutputPtr;

  ExecPackFieldCellOutputPtr PackageMaskCell;

  dax::cont::ArrayHandle<MaskType> MaskCellHandle;
  dax::cont::ArrayHandle<MaskType> MaskPointHandle;
};



} //internal
} //exec
} //dax


#endif // __dax_exec_internal_ScheduleRemoveCell_h
