/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#ifndef __PipelinePISTON_h
#define __PipelinePISTON_h

#include <dax/cont/UniformGrid.h>

class vtkImageData;

void RunPipelinePISTON(int pipeline, const dax::cont::UniformGrid &dgrid, vtkImageData* grid);

#endif //__PipelinePISTON_h
