#ifndef __dax_exec_internal_kernel_ScheduleGenerateTopology_h
#define __dax_exec_internal_kernel_ScheduleGenerateTopology_h

#include <dax/Types.h>

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

  DAX_EXEC_EXPORT void operator()(dax::Id index) const
  {
    typedef typename MaskPortalType::ValueType MaskType;
    dax::exec::internal::FieldSet(this->OutMask,
                                  index,
                                  static_cast<MaskType>(0),
                                  this->ErrorMessage);
  }

  DAX_CONT_EXPORT void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->ErrorMessage = errorMessage;
  }

private:
  MaskPortalType OutMask;
  dax::exec::internal::ErrorMessageBuffer ErrorMessage;
};

template<class MaskPortalType>
struct GetUsedPointsFunctor
{
  DAX_CONT_EXPORT
  GetUsedPointsFunctor(const MaskPortalType &outMask)
    : OutMask(outMask) {  }

  DAX_EXEC_EXPORT void operator()(dax::Id daxNotUsed(key),
                                  dax::Id value) const
  {
    typedef typename MaskPortalType::ValueType MaskType;
    dax::exec::internal::FieldSet(this->OutMask,
                                  value,
                                  static_cast<MaskType>(1),
                                  this->ErrorMessage);
  }

  DAX_CONT_EXPORT void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->ErrorMessage = errorMessage;
  }

private:
  MaskPortalType OutMask;
  dax::exec::internal::ErrorMessageBuffer ErrorMessage;
};


template<class ArrayPortalType>
struct LowerBoundsInputFunctor
{
  DAX_CONT_EXPORT LowerBoundsInputFunctor(const ArrayPortalType &array)
    : Array(array) {  }

  DAX_EXEC_EXPORT void operator()(dax::Id index) const
  {
    dax::exec::internal::FieldSet(this->Array,
                                  index,
                                  index + 1,
                                  this->ErrorMessage);
  }

  DAX_CONT_EXPORT void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->ErrorMessage = errorMessage;
  }

private:
  ArrayPortalType Array;
  dax::exec::internal::ErrorMessageBuffer ErrorMessage;
};



template<class WorkletType, class InputTopologyType, class OutputTopologyType>
struct GenerateTopologyFunctor
{
  typedef typename InputTopologyType::CellType InputCellType;
  typedef typename OutputTopologyType::CellType OutputCellType;

  DAX_CONT_EXPORT GenerateTopologyFunctor(
      WorkletType &worklet,
      const InputTopologyType &inputTopology,
      const OutputTopologyType &outputTopology)
    : Worklet(worklet),
      InputTopology(inputTopology),
      OutputTopology(outputTopology) {  }

  DAX_EXEC_EXPORT void operator()(dax::Id outputCellIndex,
                                  dax::Id inputCellIndex) const
  {
    InputCellType inputCell(this->InputTopology, inputCellIndex);
    typename OutputCellType::PointConnectionsType outputCellConnections;

    this->Worklet(inputCell, outputCellConnections);

    // Write cell connections back to cell array.
    dax::Id index = outputCellIndex * OutputCellType::NUM_POINTS;
    for (int localIndex = 0;
         localIndex < OutputCellType::NUM_POINTS;
         localIndex++)
      {
      // This only actually works if OutputTopologyType is TopologyUnstructured.
      dax::exec::internal::FieldSet(this->OutputTopology.CellConnections,
                                    index,
                                    outputCellConnections[localIndex],
                                    this->Worklet);
      index++;
      }
  }

  DAX_CONT_EXPORT void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
  }

private:
  WorkletType Worklet;
  InputTopologyType InputTopology;
  OutputTopologyType OutputTopology;
};

}
}
}
} //dax::exec::internal::kernel


#endif // SCHEDULEGENERATETOPOLOGY_H
