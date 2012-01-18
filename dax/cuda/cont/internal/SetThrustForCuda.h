/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_internal_SetThrustForCuda_h
#define __dax_cuda_cont_internal_SetThrustForCuda_h

#ifndef THRUST_DEVICE_BACKEND
#define THRUST_DEVICE_BACKEND THRUST_DEVICE_BACKEND_CUDA
#else // defined THRUST_DEVICE_BACKEND
#if THRUST_DEVICE_BACKEND != THRUST_DEVICE_BACKEND_CUDA
#error Thrust device backend set incorrectly.
#endif // THRUST_DEVICE_BACKEND != THRUST_DEVICE_BACKEND_CUDA
#endif // defined THRUST_DEVICE_BACKEND

#endif //__dax_cuda_cont_internal_SetThrustForCuda_h
