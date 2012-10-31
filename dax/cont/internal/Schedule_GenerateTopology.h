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

//This is used to generate the implict VisitIndex values.
//If GenerateVisitIndex is equal to a boost false type, we
template<bool GenerateVisitIndex,
         typename Algorithm,
         typename HandleType>
struct MakeVisitIndexControlType
{
  typedef HandleType Type;

  template<typename Scheduler, typename OtherHandleType>
  void operator()(Scheduler &scheduler,
                  const OtherHandleType& inputCellIds, Type& visitIndices) const
  {
    //To determine the number of times we have already visited
    //the current input cell, we take the lower bounds of the
    //input cell id array. The resulting number subtracted from the WorkId
    //gives us the number of times we have visited that cell
    visitIndices.PrepareForOutput(inputCellIds.GetNumberOfValues());
    Algorithm::LowerBounds(inputCellIds, inputCellIds, visitIndices);
    scheduler.Invoke(dax::exec::internal::kernel::ComputeVisitIndex(),
                     visitIndices);
  }
};

template<typename Algorithm, typename HandleType>
struct MakeVisitIndexControlType<false,Algorithm,HandleType>
{
  typedef int Type;

  template<typename Scheduler, typename OtherHandleType>
  void operator()(Scheduler &scheduler,
                  const OtherHandleType& daxNotUsed(handle), Type& value) const
  {
    //The visitIndex is not requested, so we fill in the control side argument
    //with a integer value that will be parsed by the bindings code, but
    //won't be uploaded to the execution env.
    value = 0;
  }
};

template<class InGridType, class OutGridType, typename MaskType>
void FillPointMask(const InGridType &inGrid,
                       const OutGridType &outGrid,
                       MaskType mask) const
  {
    typedef typename MaskType::PortalExecution MaskPortalType;

    // Clear out the mask, have to allocate the size first
    // so that  works properly
    mask.PrepareForOutput(inGrid.GetNumberOfPoints());

    this->Invoke(dax::exec::internal::kernel::ClearUsedPointsFunctor(),
             mask);

    // Mark every point that is used at least once.
    // This only works when outGrid is an UnstructuredGrid.
    this->Invoke(dax::exec::internal::kernel::GetUsedPointsFunctor(),
             dax::cont::make_MapAdapter(outGrid.GetCellConnections(),
             mask,
             inGrid.GetNumberOfPoints()));
  }

template<typename InGridType,typename OutGridType, typename MaskType>
void RemoveDuplicatePoints(const InGridType &inGrid,
                        OutGridType& outGrid,
                        MaskType const mask ) const
  {
    // Here we are assuming OutGridType is an UnstructuredGrid so that we
    // can set point and connectivity information.

    typedef dax::cont::internal::DeviceAdapterAlgorithm<DeviceAdapterTag>
        Algorithm;

    //extract the point coordinates that we need for the new topology
    Algorithm::StreamCompact(inGrid.GetPointCoordinates(),
                             mask,
                             outGrid.GetPointCoordinates());

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
    Algorithm::Copy(outGrid.GetCellConnections(), usedPointIndices);
    Algorithm::Sort(usedPointIndices);
    Algorithm::Unique(usedPointIndices);
    // Modify the connections of outGrid to point to compacted points.
    Algorithm::LowerBounds(usedPointIndices, outGrid.GetCellConnections());
    }
  }

//want the basic implementation to be easily edited, instead of inside
//the BOOST_PP block and unreadable
template <typename WorkType,
          typename InputGrid,
          typename OutputGrid>
void GenerateNewTopology(
    dax::cont::ScheduleGenerateTopology<WorkType,DeviceAdapterTag>& newTopo,
    const InputGrid& inputGrid,
    OutputGrid& outputGrid) const
  {
  typedef dax::cont::internal::DeviceAdapterAlgorithm<DeviceAdapterTag>
      Algorithm;
  typedef dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTagBasic,
      DeviceAdapterTag> IdArrayHandleType;

  //do an inclusive scan of the cell count / cell mask to get the number
  //of cells in the output
  IdArrayHandleType scannedNewCellCounts;
  const dax::Id numNewCells =
      Algorithm::ScanInclusive(newTopo.GetClassification(),
                               scannedNewCellCounts);

  if(newTopo.GetReleaseClassification())
    {
    newTopo.DoReleaseClassification();
    }

  //fill the validCellRange with the values from 1 to size+1, this is used
  //for the lower bounds to compute the right indices
  IdArrayHandleType validCellRange;
  validCellRange.PrepareForOutput(numNewCells);
  this->Invoke(dax::exec::internal::kernel::Index(),
                  validCellRange);

  //now do the lower bounds of the cell indices so that we figure out
  //which original topology indexs match the new indices.
  Algorithm::UpperBounds(scannedNewCellCounts, validCellRange);

  // We are done with scannedNewCellCounts.
  scannedNewCellCounts.ReleaseResources();

  //we get our magic here. we need to wrap some paramemters and pass
  //them to the real scheduler
  this->Invoke(newTopo.GetWorklet(),
                   dax::cont::make_MapAdapter(validCellRange,inputGrid,
                                             inputGrid.GetNumberOfCells()),
                   outputGrid);
  //call this here as we have stripped out the input and output grids
  if(newTopo.GetRemoveDuplicatePoints())
    {
    this->FillPointMask(inputGrid,outputGrid, newTopo.GetPointMask());
    this->RemoveDuplicatePoints(inputGrid,outputGrid, newTopo.GetPointMask());
    }
  }
public:
#else
# if _dax_pp_sizeof___T > 1
//only expose this for 2 or more parameters, since you can't
//generate topology without input and output params
public:
template <typename WorkType, _dax_pp_typename___T>
void Invoke(
    dax::cont::ScheduleGenerateTopology<WorkType,DeviceAdapterTag> newTopo,
     _dax_pp_params___(a))
  {
  this->GenerateNewTopology(newTopo,_dax_pp_args___(a));
  }

# define   SIZE                   _dax_pp_sizeof___T
# define  _SGT_pp_typename___T    BOOST_PP_ENUM_SHIFTED_PARAMS (SIZE, typename T___)
# define  _SGT_pp_params___(x)    BOOST_PP_ENUM_SHIFTED_BINARY_PARAMS(SIZE, T___, x)
# define  _SGT_pp_args___(x)      BOOST_PP_ENUM_SHIFTED_PARAMS(SIZE, x)

template <typename WorkType,
          typename InputGrid,
          typename OutputGrid,
          _SGT_pp_typename___T>
void GenerateNewTopology(
    dax::cont::ScheduleGenerateTopology<WorkType,DeviceAdapterTag>& newTopo,
    const InputGrid& inputGrid,
    OutputGrid& outputGrid,
    _SGT_pp_params___(a)) const
  {
  typedef dax::cont::internal::DeviceAdapterAlgorithm<DeviceAdapterTag>
      Algorithm;
  typedef dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTagBasic,
      DeviceAdapterTag> IdArrayHandleType;

  //do an inclusive scan of the cell count / cell mask to get the number
  //of cells in the output
  IdArrayHandleType scannedNewCellCounts;
  const dax::Id numNewCells =
      Algorithm::ScanInclusive(newTopo.GetClassification(),
                               scannedNewCellCounts);

  if(newTopo.GetReleaseClassification())
    {
    newTopo.DoReleaseClassification();
    }

  //fill the validCellRange with the values from 1 to size+1, this is used
  //for the lower bounds to compute the right indices
  IdArrayHandleType validCellRange;
  validCellRange.PrepareForOutput(numNewCells);
   this->Invoke(dax::exec::internal::kernel::Index(),
                    validCellRange);

  //now do the lower bounds of the cell indices so that we figure out
  //which original topology indexs match the new indices.
  Algorithm::UpperBounds(scannedNewCellCounts, validCellRange);

  // We are done with scannedNewCellCounts.
  scannedNewCellCounts.ReleaseResources();

  //we need to scan the args of the generate topology worklet
  //and determine if we have the VisitIndex signature. If we do,
  //we have to call a different Invoke algorithm, which properly uploads
  //the visitIndex information. Since this information is slow to compute we don't
  //want to always upload the information, instead we want to see if the user
  //has requested it.
  typedef dax::internal::ReplaceAndExtendSignatures<
              WorkType,
              dax::cont::sig::VisitIndex,
              dax::cont::sig::to_placeholder,
              dax::cont::arg::Field>  ModifiedWorkletSignatures;

  typedef typename ModifiedWorkletSignatures::found VisitIndexFound;

  typedef MakeVisitIndexControlType<VisitIndexFound::value,
              Algorithm,
              IdArrayHandleType> VisitContFunction;
  typedef typename VisitContFunction::Type VisitContType;

  //generate the visitIndex, if we don't need it we will create a dummy
  //control side signature argument that is just a constant value dax::Id
  VisitContType visitIndex;
  VisitContFunction()(*this,validCellRange,visitIndex);

  //now that we have index generated, we have to build the new worklet
  //that has the updated signature
  typedef typename dax::internal::BuildSignature<
            typename ModifiedWorkletSignatures::ControlSignature>::type NewContSig;
  typedef typename dax::internal::BuildSignature<
            typename ModifiedWorkletSignatures::ExecutionSignature>::type NewExecSig;

  //make sure to pass the user worklet down to the new derived worklet, so
  //that we get any member variable values that they have set
  dax::exec::internal::kernel::DerivedWorklet<WorkType,NewContSig,NewExecSig>
    derivedWorklet(newTopo.GetWorklet());

  //we get our magic here. we need to wrap some paramemters and pass
  //them to the real scheduler. The visitIndex must be last, as that is the
  //hardcoded location the ReplaceAndExtendSignatures will place it at
  this->Invoke(derivedWorklet,
                   dax::cont::make_MapAdapter(validCellRange,inputGrid,
                                             inputGrid.GetNumberOfCells()),
                   outputGrid,
                  _SGT_pp_args___(a),
                  visitIndex);
  //call this here as we have stripped out the input and output grids
  if(newTopo.GetRemoveDuplicatePoints())
    {
    this->FillPointMask(inputGrid,outputGrid, newTopo.GetPointMask());
    this->RemoveDuplicatePoints(inputGrid,outputGrid, newTopo.GetPointMask());
    }
  }
#endif //_dax_pp_sizeof___T > 1


#endif // defined(BOOST_PP_IS_ITERATING)

