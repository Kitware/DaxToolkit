/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#ifndef __PipelineDebug_h
#define __PipelineDebug_h

class vtkImageData;

void RunPipelineVTK(int pipeline, vtkImageData* grid);

#endif //__PipelineDebug_h
