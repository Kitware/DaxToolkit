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
#include <dax/cont/internal/DeviceAdapterTag.h>
#include <dax/exec/WorkletInterpolatedCell.h>
#include <dax/internal/ParameterPack.h>

#include <dax/math/Compare.h>
#include <dax/cont/dispatcher/AddVisitIndexArg.h>
#include <dax/exec/internal/kernel/GenerateWorklets.h>

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

  typedef dax::cont::ArrayHandle< dax::Vector3,
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


  template<typename T, typename Container1,
           typename Container2, typename DeviceAdapter>
  DAX_CONT_EXPORT
  bool CompactPointField(
      const dax::cont::ArrayHandle<T,Container1,DeviceAdapter>& input,
      dax::cont::ArrayHandle<T,Container2,DeviceAdapter>& output)
    {
    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>
                                        Algorithm;
    typedef typename dax::cont::ArrayHandle<T,Container1,DeviceAdapter>::
                                        PortalConstExecution InPortalType;
    typedef typename InterpolationWeightsType::
                                        PortalConstExecution InterpPortalType;
    typedef typename dax::cont::ArrayHandle<T,Container2,DeviceAdapter>::
                                        PortalExecution OutPortalType;

    const dax::Id size = this->InterpolationWeights.GetNumberOfValues();
    dax::exec::internal::kernel::InterpolateFieldToField<InterpPortalType,
                                                         InPortalType,
                                                         OutPortalType>
        interpolate( InterpolationWeights.PrepareForInput(),
                     input.PrepareForInput(),
                     output.PrepareForOutput(size));

    Algorithm::Schedule(interpolate, size);

    //after each time we interpolate we unload the interpolation weights
    //from the execution env so that we don't hold reserve exec memory
    //longer than needed
    this->InterpolationWeights.GetPortalControl(); //copy to cont
    this->InterpolationWeights.ReleaseResourcesExecution();

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

    //make fake indicies so that the worklet can write out the interpolated
    //cell points information as geometry
    outputGrid.GetCellConnections().PrepareForOutput(
       numNewCells * dax::CellTraits<typename OutputGrid::CellTag>::NUM_VERTICES);
    dax::cont::DispatcherMapField< dax::exec::internal::kernel::Index >()
                                      .Invoke(outputGrid.GetCellConnections());

    //Next step is to set the dispatcher to fill the output geometry
    //with the interpolated cell values. We use the outputGrid,
    //since a vec3 is what the interpolated cell values and the outputGrids
    //coordinate space are

    //we get our magic here. we need to wrap some parameters and pass
    //them to the real dispatcher
    DerivedWorkletType derivedWorklet(worklet);

    this->BasicInvoke( derivedWorklet,
          arguments.template Replace<1>(
            dax::cont::make_Permutation(validCellRange,inputGrid,
                                        inputGrid.GetNumberOfCells()))
          .Append(visitIndex));

    //now that the interpolated grid is filled we now have to properly
    //fixup the topology and coordinates
    this->ResolveCoordinates(inputGrid,outputGrid,
                             this->GetRemoveDuplicatePoints());
  }

  //take the input grid and the interpolated grid to produce the new points
  //that fill the output grid. In the future the user should be able to to
  //specify the coordinate array to interpolate on, instead of it being based
  //on the input grid. This would allow us to do some smarter contouring on
  //moving coordinate fields, where the count doesn't change
  template <typename InputGrid, typename OutputGrid>
  DAX_CONT_EXPORT void ResolveCoordinates(const InputGrid& inputGrid,
                                          OutputGrid& outputGrid,
                                          bool removeDuplicates )
  {
    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>
        Algorithm;

    //we have stored the interpolation weights inside the output grid coordinates
    //which in the future needs to be changed. So store a copy of them
    //into our own array handle
    Algorithm::Copy(outputGrid.GetPointCoordinates(),
                    this->InterpolationWeights);

    if(removeDuplicates)
      {
      // the sort and unique will get us the subset of new points
      // the lower bounds on the subset and the original coords, will produce
      // the resulting topology array

      dax::math::SortLess comparisonFunctor;

      Algorithm::Sort(this->InterpolationWeights, comparisonFunctor );
      Algorithm::Unique(this->InterpolationWeights);
      Algorithm::LowerBounds(this->InterpolationWeights,
                             outputGrid.GetPointCoordinates(),
                             outputGrid.GetCellConnections(),
                             comparisonFunctor );

      //reduce and resize outputGrid
      Algorithm::Copy(this->InterpolationWeights,
                      outputGrid.GetPointCoordinates());
      }

    this->CompactPointField(inputGrid.GetPointCoordinates(),
                            outputGrid.GetPointCoordinates());
  }


  bool RemoveDuplicatePoints;
  bool ReleaseCount;
  CountHandleType Count;

  InterpolationWeightsType InterpolationWeights;

};

} }

#endif //__dax_cont_dispatcher_GenerateInterpolatedCells_h

