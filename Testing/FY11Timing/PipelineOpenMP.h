/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#ifndef __PipelineOpenMP_h
#define __PipelineOpenMP_h

#include <dax/cont/UniformGrid.h>

void RunPipelineOpenMP(int pipeline, const dax::cont::UniformGrid &grid);

#endif //__PipelineOpenMP_h
