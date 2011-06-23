
#ifndef __DaxExecutionEnvironment_h
#define __DaxExecutionEnvironment_h

#include "DaxArray.cu"
#include "DaxArrayIrregular.cu"
#include "DaxArrayStructuredConnectivity.cu"
#include "DaxArrayStructuredPoints.cu"
#include "DaxArrayTraits.cu"
#include "DaxCell.cu"
#include "DaxCellTypes.h"
#include "DaxCommon.h"
#include "DaxDataObject.cu"
#include "DaxExecutionEnvironment.h"
#include "DaxField.cu"
#include "DaxWork.cu"

#define DAX_WORKLET __device__
//#define DAX_IN const
#define DAX_IN 
#define DAX_OUT

#endif
