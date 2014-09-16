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
#ifndef __dax_cont_DispatcherGenerateInterpolatedCells_h
#define __dax_cont_DispatcherGenerateInterpolatedCells_h

#include <dax/Types.h>

#include <dax/cont/dispatcher/DispatcherBase.h>

#include <dax/cont/dispatcher/AddVisitIndexArg.h>
#include <dax/cont/internal/DeviceAdapterTag.h>
#include <dax/cont/internal/EdgeInterpolatedGrid.h>
#include <dax/exec/internal/kernel/GenerateWorklets.h>
#include <dax/exec/WorkletInterpolatedCell.h>
#include <dax/internal/ParameterPack.h>
#include <dax/math/Compare.h>

namespace dax { namespace cont {


template <
  class WorkletType_,
  class CountHandleType_ = dax::cont::ArrayHandle< dax::Id >,
  class DeviceAdapterTag_ = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class DispatcherGenerateInterpolatedCells :
  public dax::cont::dispatcher::DispatcherBase<
          DispatcherGenerateInterpolatedCells< WorkletType_, CountHandleType_, DeviceAdapterTag_ >,
          dax::exec::WorkletInterpolatedCell,
          WorkletType_,
          DeviceAdapterTag_ >
{

  typedef dax::cont::dispatcher::DispatcherBase< DispatcherGenerateInterpolatedCells< WorkletType_, CountHandleType_,  DeviceAdapterTag_>,
                                                 dax::exec::WorkletInterpolatedCell,
                                                 WorkletType_,
                                                 DeviceAdapterTag_> Superclass;
  friend class dax::cont::dispatcher::DispatcherBase< DispatcherGenerateInterpolatedCells< WorkletType_, CountHandleType_, DeviceAdapterTag_>,
                                                 dax::exec::WorkletInterpolatedCell,
                                                 WorkletType_,
                                                 DeviceAdapterTag_>;

  typedef dax::cont::ArrayHandle< dax::PointAsEdgeInterpolation,
                  dax::cont::ArrayContainerControlTagBasic,
                  DeviceAdapterTag_ >  InterpolationWeightsType;
public:
  typedef WorkletType_ WorkletType;
  typedef CountHandleType_ CountHandleType;
  typedef DeviceAdapterTag_ DeviceAdapterTag;


  DAX_CONT_EXPORT
  DispatcherGenerateInterpolatedCells(const CountHandleType &count):
    Superclass( WorkletType() ),
    RemoveDuplicatePoints(true),
    ReleaseCount(true),
    Count(count),
    InterpolationWeights()
    { }

  DAX_CONT_EXPORT
  DispatcherGenerateInterpolatedCells(const CountHandleType &count,
                            const WorkletType& work):
    Superclass( work ),
    RemoveDuplicatePoints(true),
    ReleaseCount(true),
    Count(count),
    InterpolationWeights()
    { }


  DAX_CONT_EXPORT void SetReleaseCount(bool b)
    { ReleaseCount = b; }

  DAX_CONT_EXPORT bool GetReleaseCount() const
  { return ReleaseCount; }

  DAX_CONT_EXPORT
  CountHandleType GetCount() const
    { return Count; }

  DAX_CONT_EXPORT
  void DoReleaseCount()
    { Count.ReleaseResourcesExecution(); }

  DAX_CONT_EXPORT
  void SetRemoveDuplicatePoints(bool b)
    { RemoveDuplicatePoints = b; }

  DAX_CONT_EXPORT
  bool GetRemoveDuplicatePoints() const
    { return RemoveDuplicatePoints; }


  template<typename Container1,
           typename Container2,
           typename DeviceAdapter>
  DAX_CONT_EXPORT
  bool CompactPointField(
      const dax::cont::ArrayHandle<dax::Vector3,Container1,DeviceAdapter>& input,
      dax::cont::ArrayHandle<dax::Vector3,Container2,DeviceAdapter>& output)
    {

    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>
                                        Algorithm;

    typedef typename dax::cont::ArrayHandle<dax::Vector3,
            Container1,DeviceAdapter>::PortalConstExecution InPortalType;

    typedef typename InterpolationWeightsType::PortalConstExecution
        WeightsPortalType;

    typedef typename dax::cont::ArrayHandle<dax::Vector3,
            Container2,DeviceAdapter>::PortalExecution OutPortalType;

    //interpolation holds the proper size
    const dax::Id size = this->InterpolationWeights.GetNumberOfValues();
    dax::exec::internal::kernel::InterpolateFieldToField<InPortalType,
                                                         WeightsPortalType,
                                                         OutPortalType>
        interpolate( input.PrepareForInput(),
                     this->InterpolationWeights.PrepareForInput(),
                     output.PrepareForOutput(size));

    Algorithm::Schedule(interpolate, size);

    return true;
    }

private:

  template<typename ParameterPackType>
  DAX_CONT_EXPORT
  void DoInvoke(WorkletType worklet,
                ParameterPackType arguments)
  {
    this->GenerateNewTopology(
          worklet,
          dax::internal::ParameterPackGetArgument<1>(arguments),
          dax::internal::ParameterPackGetArgument<2>(arguments),
          arguments);
  }


  template <typename InputGrid,
            typename OutputGrid,
            typename ParameterPackType>
  DAX_CONT_EXPORT void GenerateNewTopology(
      WorkletType worklet,
      InputGrid inputGrid,
      OutputGrid outputGrid,
      const ParameterPackType &arguments)
  {
    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>
        Algorithm;
    typedef dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTagBasic,
        DeviceAdapterTag> IdArrayHandleType;

    dax::cont::internal::EdgeInterpolatedGrid<
        typename OutputGrid::CellTag,
        ArrayContainerControlTagBasic,
        ArrayContainerControlTagBasic,
        DeviceAdapterTag > edgeInterpolatedOutputGrid;

    //do an inclusive scan of the cell count / cell mask to get the number
    //of cells in the output
    IdArrayHandleType scannedNewCellCounts;
    const dax::Id numNewCells =
        Algorithm::ScanInclusive(this->GetCount(),
                                 scannedNewCellCounts);

    if(this->GetReleaseCount())
      {
      this->DoReleaseCount();
      }

    if(numNewCells == 0)
      {
      //nothing to do
      return;
      }


    //now do the uppper bounds of the cell indices so that we figure out
    //which original topology indexs match the new indices.
    IdArrayHandleType validCellRange;
    Algorithm::UpperBounds(scannedNewCellCounts,
                   dax::cont::make_ArrayHandleCounting(dax::Id(0),numNewCells),
                   validCellRange);

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
    typedef dax::cont::dispatcher::AddVisitIndexArg<WorkletType,
      Algorithm,IdArrayHandleType> AddVisitIndexFunctor;
    typedef typename AddVisitIndexFunctor::VisitIndexArgType IndexArgType;
    typedef typename AddVisitIndexFunctor::DerivedWorkletType DerivedWorkletType;

    IndexArgType visitIndex;
    AddVisitIndexFunctor createVisitIndex;
    createVisitIndex(validCellRange,visitIndex);

    //create a temporary edge interpolation unstructured grid, that has
    //the cell indices faked so that the worklet can write out the interpolated
    //cell points edge interpolation information
    edgeInterpolatedOutputGrid.GetCellConnections().PrepareForOutput(
       numNewCells * dax::CellTraits<typename OutputGrid::CellTag>::NUM_VERTICES);
    dax::cont::DispatcherMapField< dax::exec::internal::kernel::Index >()
                                      .Invoke(edgeInterpolatedOutputGrid.GetCellConnections());

    //Next step is to set the dispatcher to fill the output geometry
    //with the interpolated cell values. We use edgeInterpolatedOutputGrid,
    //since it 'points' are really interpolation of  2 edges and the weight

    //we get our magic here. we need to wrap some parameters and pass
    //them to the real dispatcher
    DerivedWorkletType derivedWorklet(worklet);

    this->BasicInvoke( derivedWorklet,
          arguments.template Replace<1>(
            dax::cont::make_Permutation(validCellRange,inputGrid,
                                        inputGrid.GetNumberOfCells()))
          .template Replace<2>( edgeInterpolatedOutputGrid )
          .Append(visitIndex));

    //now that the interpolated grid is filled we now have to properly
    //fixup the topology and coordinates
    this->ResolveCoordinates(inputGrid,
                             edgeInterpolatedOutputGrid,
                             outputGrid,
                             this->GetRemoveDuplicatePoints());
  }

  //take the input grid and the interpolated grid to produce the new points
  //that fill the output grid. In the future the user should be able to to
  //specify the coordinate array to interpolate on, instead of it being based
  //on the input grid. This would allow us to do some smarter contouring on
  //moving coordinate fields, where the count doesn't change
  template <typename InputGrid, typename InterpolatedGrid, typename OutputGrid>
  DAX_CONT_EXPORT void ResolveCoordinates(const InputGrid& inputGrid,
                                          const InterpolatedGrid& interpolatedGrid,
                                          OutputGrid& outputGrid,
                                          bool removeDuplicates )
  {
    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>
        Algorithm;

    // We need to store the interpolation weights to be able to interpolate
    // point scalar fields (including point coordinates). We also need to
    // find duplicate interpolations if we want to remove duplicate points.
    Algorithm::Copy(interpolatedGrid.GetInterpolatedPoints(),
                    this->InterpolationWeights);

    if(removeDuplicates)
      {
      // the sort and unique will get us the subset of new points
      // the lower bounds on the subset and the original coords, will produce
      // the resulting topology array
      Algorithm::Sort(this->InterpolationWeights);
      Algorithm::Unique(this->InterpolationWeights);
      Algorithm::LowerBounds(this->InterpolationWeights,
                             interpolatedGrid.GetInterpolatedPoints(),
                             outputGrid.GetCellConnections()
                             );


      this->CompactPointField(inputGrid.GetPointCoordinates(),
                              outputGrid.GetPointCoordinates());
      }
    else
      {
      this->CompactPointField(inputGrid.GetPointCoordinates(),
                              outputGrid.GetPointCoordinates());
      //we need to  copy the cells connections from the interpolatedGrid
      //over to the output grid
      Algorithm::Copy(interpolatedGrid.GetCellConnections(),
                      outputGrid.GetCellConnections());
      }
  }


  bool RemoveDuplicatePoints;
  bool ReleaseCount;
  CountHandleType Count;
  InterpolationWeightsType InterpolationWeights;

};

} }

#endif //__dax_cont_dispatcher_GenerateInterpolatedCells_h

