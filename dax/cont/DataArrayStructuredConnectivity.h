/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_DataArrayStructuredConnectivity_h
#define __dax_cont_DataArrayStructuredConnectivity_h

#include <dax/cont/DataArrayStructuredPoints.h>

namespace dax { namespace cont {

daxDeclareClass(DataArrayStructuredConnectivity);

/// DataArrayStructuredConnectivity is used for connectivity array for a
/// structured dataset.
class DataArrayStructuredConnectivity : public dax::cont::DataArrayStructuredPoints
{
public:
  DataArrayStructuredConnectivity();
  virtual ~DataArrayStructuredConnectivity();
  daxTypeMacro(DataArrayStructuredConnectivity, dax::cont::DataArrayStructuredPoints);

private:
  daxDisableCopyMacro(DataArrayStructuredConnectivity)
};

}}
#endif
