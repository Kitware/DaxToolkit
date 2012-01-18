#ifndef __dax_cont_internal_Classify_h
#define __dax_cont_internal_Classify_h

#include <dax/internal/ExportMacros.h>
#include <dax/internal/DataArray.h>

#include <dax/cont/ArrayHandle.h>


namespace dax {
namespace cont {
namespace internal {

/// Base Classify Object. The Classify class uses the CRTP to allow inhertiance.
/// All children classes need to define the following methods:
///   template<typename InputType>
///   dax::Id classifySize(const InputType& input);
///
///   template<typename InputType, typename HandleType>
///   void classifyWorklet(const InputType& input, HandleType& handle);
///
/// classifySize defines the total length of the memory required to store
/// the results of the classify step. InputType is the geometery that the mapreduce
/// algorithm is running on.
///
/// classifyWorklet defines the actual worklet that does the classify function. InputType
/// is the geometery and handle is an ArrayHandle of the storage container for the
/// classify result.


template<class Derived,
         class Parameters,
         DAX_DeviceAdapter_TP>
class Classify
{
public:
  typedef dax::Id ValueType;

  template< typename GridType, typename HandleType>
  void run(const GridType& geom, HandleType& handle)
    {
    //call the function to determine the size of the
    //classify result array. This is needed as the on a gpu
    //we can't allocate memory while in a kernel
    this->ResultSize = this->callClassifySize(geom);

    //the single size means we are making a flat array,
    //instead of having some form of a struct of arrays or zipped arrays
    this->Result = dax::cont::ArrayHandle<ValueType,DeviceAdapter>(this->ResultSize);

    this->callClassify(geom,handle);
    }

  dax::cont::ArrayHandle<ValueType>& GetResult()
    { return Result; }


  dax::Id GetResultSize()
    { return ResultSize; }

  template<typename InputType>
  dax::Id callClassifySize(const InputType& geom)
  {
    return static_cast<Derived*>(this)->Size(geom);
  }

  template<typename GridType, typename HandleType>
  void callClassify(const GridType& geom, HandleType &handle)
  {
    return static_cast<Derived*>(this)->Classify(geom,handle);
  }

private:
  dax::Id ResultSize;
  dax::cont::ArrayHandle<ValueType> Result;

};

} //internal
} //cont
} //dax


#endif // __dax_cont_internal_Classify_h
