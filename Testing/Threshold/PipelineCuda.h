/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#ifndef __PipelineCuda_h
#define __PipelineCuda_h

#include <dax/cont/UniformGrid.h>

void RunPipelineCuda(int pipeline, const dax::cont::UniformGrid &grid);

#endif //__PipelineCuda_h
