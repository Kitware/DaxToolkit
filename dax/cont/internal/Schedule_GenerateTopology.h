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

//-----------------------------------------------------------------------------
//General methods that all version of the generate topology use
#if !defined(BOOST_PP_IS_ITERATING)
private:
template<class InGridType, class OutGridType, typename MaskType>
void FillPointMask(const InGridType &inGrid,
                       const OutGridType &outGrid,
                       MaskType& mask)
  {
    typedef typename MaskType::PortalExecution MaskPortalType;

    // Clear out the mask, have to allocate the size first
    // so that  works properly
    mask.PrepareForOutput(inGrid.GetNumberOfPoints());

    this->operator()(dax::exec::internal::kernel::ClearUsedPointsFunctor(),
             mask);

    // Mark every point that is used at least once.
    // This only works when outGrid is an UnstructuredGrid.
    this->operator()(dax::exec::internal::kernel::GetUsedPointsFunctor(),
             dax::cont::make_MapAdapter(outGrid.GetCellConnections(),
             mask,
             inGrid.GetNumberOfPoints()));
  }

template<typename InGridType,typename OutGridType, typename MaskType>
void RemoveDuplicatePoints(const InGridType &inGrid,
                        OutGridType& outGrid,
                        MaskType const& mask )
  {
    // Here we are assuming OutGridType is an UnstructuredGrid so that we
    // can set point and connectivity information.

    //extract the point coordinates that we need for the new topology
    dax::cont::internal::StreamCompact(inGrid.GetPointCoordinates(),
                                       mask,
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
    typedef dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTagBasic,
        DeviceAdapterTag> IdArrayHandleType;
    IdArrayHandleType usedPointIndices;
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

//want the basic implementation to be easily edited, instead of inside
//the BOOST_PP block and unreadable
template <typename WorkType,
          typename InputGrid,
          typename OutputGrid>
void GenerateNewTopology(
    dax::cont::ScheduleGenerateTopology<WorkType,DeviceAdapterTag> newTopo,
    const InputGrid& inputGrid,
    OutputGrid& outputGrid)
  {
  typedef dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTagBasic,
      DeviceAdapterTag> IdArrayHandleType;

  //do an inclusive scan of the cell count / cell mask to get the number
  //of cells in the output
  IdArrayHandleType scannedNewCellCounts;
  const dax::Id numNewCells =
      dax::cont::internal::InclusiveScan(newTopo.GetClassification(),
                                         scannedNewCellCounts,
                                         DeviceAdapterTag());

  if(newTopo.GetReleaseClassification())
    {
    newTopo.DoReleaseClassification();
    }

  //fill the validCellRange with the values from 1 to size+1, this is used
  //for the lower bounds to compute the right indices
  IdArrayHandleType validCellRange;
  validCellRange.PrepareForOutput(numNewCells);
  this->operator()(dax::exec::internal::kernel::IndexPlusOne(),
                   validCellRange);

  //now do the lower bounds of the cell indices so that we figure out
  //which original topology indexs match the new indices.
  dax::cont::internal::LowerBounds(scannedNewCellCounts,
                                   validCellRange,
                                   DeviceAdapterTag());

  // We are done with scannedNewCellCounts.
  scannedNewCellCounts.ReleaseResources();

  //we get our magic here. we need to wrap some paramemters and pass
  //them to the real scheduler
  this->operator()(newTopo.GetWorklet(),
                   dax::cont::make_MapAdapter(validCellRange,inputGrid,
                                             inputGrid.GetNumberOfCells()),
                   outputGrid);
  //call this here as we have stripped out the input and output grids
  this->FillPointMask(inputGrid,outputGrid, newTopo.GetPointMask());
  if(newTopo.GetRemoveDuplicatePoints())
    {
    this->RemoveDuplicatePoints(inputGrid,outputGrid, newTopo.GetPointMask());
    }
  }
public:
#else
# if _dax_pp_sizeof___T > 2
//only expose this for 2 or more parameters, since you can't
//generate topology without input and output params
public:
template <typename WorkType, _dax_pp_typename___T>
void operator()(
    dax::cont::ScheduleGenerateTopology<WorkType,DeviceAdapterTag> newTopo,
     _dax_pp_params___(a))
  {
  this->GenerateNewTopology(newTopo,_dax_pp_args___(a));
  }
# endif // _dax_pp_sizeof___T > 2

# if _dax_pp_sizeof___T > 3
# define   SIZE                   BOOST_PP_SUB(_dax_pp_sizeof___T,1)
# define  _SGT_pp_typename___T    BOOST_PP_ENUM_SHIFTED_PARAMS (SIZE, typename T___)
# define  _SGT_pp_params___(x)    BOOST_PP_ENUM_SHIFTED_BINARY_PARAMS(SIZE, T___, x)
# define  _SGT_pp_args___(x)      BOOST_PP_ENUM_SHIFTED_PARAMS(SIZE, x)

template <typename WorkType,
          typename InputGrid,
          typename OutputGrid,
          _SGT_pp_typename___T>
void GenerateNewTopology(
    dax::cont::ScheduleGenerateTopology<WorkType,DeviceAdapterTag> newTopo,
    const InputGrid& inputGrid,
    OutputGrid& outputGrid,
    _SGT_pp_params___(a))
  {
  typedef dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTagBasic,
      DeviceAdapterTag> IdArrayHandleType;

  //do an inclusive scan of the cell count / cell mask to get the number
  //of cells in the output
  IdArrayHandleType scannedNewCellCounts;
  const dax::Id numNewCells =
      dax::cont::internal::InclusiveScan(newTopo.GetClassification(),
                                         scannedNewCellCounts,
                                         DeviceAdapterTag());

  if(newTopo.GetReleaseClassification())
    {
    newTopo.DoReleaseClassification();
    }

  //fill the validCellRange with the values from 1 to size+1, this is used
  //for the lower bounds to compute the right indices
  IdArrayHandleType validCellRange;
  validCellRange.PrepareForOutput(numNewCells);
  this->operator()(dax::exec::internal::kernel::IndexPlusOne(),
                   validCellRange);

  //now do the lower bounds of the cell indices so that we figure out
  //which original topology indexs match the new indices.
  dax::cont::internal::LowerBounds(scannedNewCellCounts,
                                   validCellRange,
                                   DeviceAdapterTag());

  // We are done with scannedNewCellCounts.
  scannedNewCellCounts.ReleaseResources();

  //we get our magic here. we need to wrap some paramemters and pass
  //them to the real scheduler
  this->operator()(newTopo.GetWorklet(),
                   dax::cont::make_MapAdapter(validCellRange,inputGrid,
                                             inputGrid.GetNumberOfCells()),
                   outputGrid,
                  _SGT_pp_args___(a));
  //call this here as we have stripped out the input and output grids
  this->FillPointMask(inputGrid,outputGrid, newTopo.GetPointMask());
  if(newTopo.GetRemoveDuplicatePoints())
    {
    this->RemoveDuplicatePoints(inputGrid,outputGrid, newTopo.GetPointMask());
    }
  }
#endif //_dax_pp_sizeof___T > 3


#endif // defined(BOOST_PP_IS_ITERATING)

