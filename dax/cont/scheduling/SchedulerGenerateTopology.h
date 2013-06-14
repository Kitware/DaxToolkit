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
//===========================================x==================================
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef __dax_cont_scheduling_SchedulerGenerateTopology_h
#define __dax_cont_scheduling_SchedulerGenerateTopology_h

#include <dax/Types.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/sig/VisitIndex.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/scheduling/SchedulerTags.h>
#include <dax/cont/scheduling/SchedulerDefault.h>
#include <dax/cont/scheduling/VerifyUserArgLength.h>
#include <dax/cont/scheduling/AddVisitIndexArg.h>

#include <dax/exec/internal/kernel/GenerateWorklets.h>

#if !(__cplusplus >= 201103L)
# include <dax/internal/ParameterPackCxx03.h>
#endif // !(__cplusplus >= 201103L)

namespace dax { namespace cont { namespace scheduling {

template <class DeviceAdapterTag>
class Scheduler<DeviceAdapterTag,dax::cont::scheduling::GenerateTopologyTag>
{
public:
  //default constructor so we can insantiate const schedulers
  DAX_CONT_EXPORT Scheduler():DefaultScheduler(){}

  //copy constructor so that people can pass schedulers around by value
  DAX_CONT_EXPORT Scheduler(
      const Scheduler<DeviceAdapterTag,
          dax::cont::scheduling::GenerateTopologyTag>& other ):
  DefaultScheduler(other.DefaultScheduler)
  {
  }

#if __cplusplus >= 201103L
  template <class WorkletType, _dax_pp_typename___T>
  DAX_CONT_EXPORT void Invoke(WorkletType w, T...a)
  {
  typedef dax::cont::scheduling::VerifyUserArgLength<WorkletType,
              sizeof...(T)> WorkletUserArgs;
  //if you are getting this error you are passing less arguments than requested
  //in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

  //if you are getting this error you are passing too many arguments
  //than requested in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

  this->GenerateNewTopology(w,T...);
  }

  //todo implement the GenerateNewTopology method with C11 syntax

#else
# define BOOST_PP_ITERATION_PARAMS_1 (3, (2, 10, <dax/cont/scheduling/SchedulerGenerateTopology.h>))
# include BOOST_PP_ITERATE()
#endif

private:

//want the basic implementation to be easily edited, instead of inside
//the BOOST_PP block and unreadable. This version of GenerateNewTopology
//handles the use case no parameters
template <class WorkletType,
          typename ClassifyHandleType,
          typename InputGrid,
          typename OutputGrid>
DAX_CONT_EXPORT void GenerateNewTopology(
    dax::cont::GenerateTopology<
    WorkletType,ClassifyHandleType>& newTopo,
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

  if(numNewCells == 0)
    {
    //nothing to do
    return;
    }

  //fill the validCellRange with the values from 1 to size+1, this is used
  //for the lower bounds to compute the right indices
  IdArrayHandleType validCellRange;
  validCellRange.PrepareForOutput(numNewCells);
  this->DefaultScheduler.Invoke(dax::exec::internal::kernel::Index(),
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
  //want to always upload the information, instead only compute when explicitly
  //requested

  //The AddVisitIndexArg does all this, plus creates a derived worklet
  //from the users worklet with the visit index added to the signature.
  typedef dax::cont::scheduling::AddVisitIndexArg<WorkletType,
    Algorithm,IdArrayHandleType> AddVisitIndexFunctor;
  typedef typename AddVisitIndexFunctor::VisitIndexArgType IndexArgType;
  typedef typename AddVisitIndexFunctor::DerivedWorkletType DerivedWorkletType;

  IndexArgType visitIndex;
  AddVisitIndexFunctor createVisitIndex;
  createVisitIndex(this->DefaultScheduler,validCellRange,visitIndex);

  DerivedWorkletType derivedWorklet(newTopo.GetWorklet());

  //we get our magic here. we need to wrap some paramemters and pass
  //them to the real scheduler
  this->DefaultScheduler.Invoke(derivedWorklet,
                   dax::cont::make_Permutation(validCellRange,inputGrid,
                                             inputGrid.GetNumberOfCells()),
                   outputGrid,
                   visitIndex);
  //call this here as we have stripped out the input and output grids
  if(newTopo.GetRemoveDuplicatePoints())
    {
    this->FillPointMask(inputGrid,outputGrid, newTopo.GetPointMask());
    this->RemoveDuplicatePoints(inputGrid,outputGrid, newTopo.GetPointMask());
    }
  }

template<class InGridType, class OutGridType, typename MaskType>
DAX_CONT_EXPORT void FillPointMask(const InGridType &inGrid,
                       const OutGridType &outGrid,
                       MaskType mask) const
  {
  typedef typename MaskType::PortalExecution MaskPortalType;

  // Clear out the mask, have to allocate the size first
  // so that  works properly
  mask.PrepareForOutput(inGrid.GetNumberOfPoints());

  this->DefaultScheduler.Invoke(dax::exec::internal::kernel::ClearUsedPointsFunctor(),
           mask);

  // Mark every point that is used at least once.
  // This only works when outGrid is an UnstructuredGrid.
  this->DefaultScheduler.Invoke(dax::exec::internal::kernel::GetUsedPointsFunctor(),
           dax::cont::make_Permutation(outGrid.GetCellConnections(),
           mask,
           inGrid.GetNumberOfPoints()));
  }

template<typename InGridType,typename OutGridType, typename MaskType>
DAX_CONT_EXPORT void RemoveDuplicatePoints(const InGridType &inGrid,
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

  typedef dax::cont::scheduling::Scheduler<DeviceAdapterTag,
    dax::cont::scheduling::ScheduleDefaultTag> SchedulerDefaultType;
  const SchedulerDefaultType DefaultScheduler;

};

} } }

#endif //__dax_cont_scheduling_GenerateTopology_h

#else // defined(BOOST_PP_IS_ITERATING)
public: //needed so that each iteration of invoke is public
template <class WorkletType, _dax_pp_typename___T>
DAX_CONT_EXPORT void Invoke(WorkletType w, _dax_pp_params___(a)) const
  {
  //we are being passed dax::cont::GenerateTopology,
  //we want the actual exec worklet that is being passed to scheduleGenerateTopo
  typedef typename WorkletType::WorkletType RealWorkletType;
  typedef dax::cont::scheduling::VerifyUserArgLength<RealWorkletType,
              _dax_pp_sizeof___T> WorkletUserArgs;
  //if you are getting this error you are passing less arguments than requested
  //in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

  //if you are getting this error you are passing too many arguments
  //than requested in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

  this->GenerateNewTopology(w,_dax_pp_args___(a));
  }

private:
template <class WorkletType,
          typename ClassifyHandleType,
          typename InputGrid,
          typename OutputGrid,
          _dax_pp_typename___T>
DAX_CONT_EXPORT void GenerateNewTopology(
    dax::cont::GenerateTopology<
    WorkletType, ClassifyHandleType >& newTopo,
    const InputGrid& inputGrid,
    OutputGrid& outputGrid,
    _dax_pp_params___(a)) const
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

  if(numNewCells == 0)
    {
    //nothing to do
    return;
    }

  //fill the validCellRange with the values from 0 to size, this is used
  //for the upper bounds to compute the right indices
  IdArrayHandleType validCellRange;
  validCellRange.PrepareForOutput(numNewCells);
  this->DefaultScheduler.Invoke(dax::exec::internal::kernel::Index(),
                    validCellRange);

  //now do the uppper bounds of the cell indices so that we figure out
  //which original topology indexs match the new indices.
  Algorithm::UpperBounds(scannedNewCellCounts, validCellRange);

  // We are done with scannedNewCellCounts.
  scannedNewCellCounts.ReleaseResources();

  //we need to scan the args of the generate topology worklet
  //and determine if we have the VisitIndex signature. If we do,
  //we have to call a different Invoke algorithm, which properly uploads
  //the visitIndex information. Since this information is slow to compute we don't
  //want to always upload the information, instead only compute when explicitly
  //requested

  //The AddVisitIndexArg does all this, plus creates a derived worklet
  //from the users worklet with the visit index added to the signature.
  typedef dax::cont::scheduling::AddVisitIndexArg<WorkletType,
    Algorithm,IdArrayHandleType> AddVisitIndexFunctor;
  typedef typename AddVisitIndexFunctor::VisitIndexArgType IndexArgType;
  typedef typename AddVisitIndexFunctor::DerivedWorkletType DerivedWorkletType;

  IndexArgType visitIndex;
  AddVisitIndexFunctor createVisitIndex;
  createVisitIndex(this->DefaultScheduler,validCellRange,visitIndex);

  DerivedWorkletType derivedWorklet(newTopo.GetWorklet());

  //we get our magic here. we need to wrap some paramemters and pass
  //them to the real scheduler. The visitIndex must be last, as that is the
  //hardcoded location the ReplaceAndExtendSignatures will place it at
  this->DefaultScheduler.Invoke(derivedWorklet,
                   dax::cont::make_Permutation(validCellRange,inputGrid,
                                             inputGrid.GetNumberOfCells()),
                   outputGrid,
                  _dax_pp_args___(a),
                  visitIndex);
  //call this here as we have stripped out the input and output grids
  if(newTopo.GetRemoveDuplicatePoints())
    {
    this->FillPointMask(inputGrid,outputGrid, newTopo.GetPointMask());
    this->RemoveDuplicatePoints(inputGrid,outputGrid, newTopo.GetPointMask());
    }
  }
#endif // defined(BOOST_PP_IS_ITERATING)
