/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxKernelArgument_h
#define __DaxKernelArgument_h

class DaxDataArray;
class DaxDataSet;

/// DaxKernelArgument is passed to the entry-point into the execution
/// environment. It encapsulates all information about the datasets used in the
/// pipeline.
class DaxKernelArgument
{
public:
  DaxDataArray* Arrays;
  DaxDataSet* Datasets;
  int NumberOfArrays;
  int NumberOfDatasets;
};

#endif
