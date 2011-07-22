/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxKernelArgument_h
#define __DaxKernelArgument_h

class DaxDataArray;
class DaxDataSet;

class DaxKernelArgument
{
public:
  DaxDataArray* Arrays;
  DaxDataSet* Datasets;
  int NumberOfArrays;
  int NumberOfDatasets;
};

#endif
