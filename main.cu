/*
 * Author:      Hampton Ford
 * Github:      @shford
 * Status:      Incomplete
 *
 * Notes:       Max file size set at 4GiB b/c of Crypto Library limitations
 *
 * License:
 *
 *
 * CUDA_Driver_Version:                         11.7
 * CUDA Capability Major/Minor version number:  8.6
 *
 * Performance:
 * ~todo time
 *
 */

#include "tasks.cuh"

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

/*
 * set shared memory capacity to 0 - cudaFuncSetAttribute(kernel_name, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
 *
 * todo V5:
 *  Multi-Thread/Multi-Process - Initial & Final File I/O
 *  Multi-Thread/Multi-Process - Task 2
 *  Replace calloc inside read_file() w/ cudaMallocHost() to pin memory
 *  Replace size w/ strlen calls in host b/c no need to transfer that
 *  In kernel change from byte-by-byte copying to memcpyasync or at least by multiples of largest primitive
 */


int main()
{
    #if VALID_USER_INPUTS
        int devices = 0;
        cudaGetDeviceCount(&devices);

        if (devices < 1) {
            printf("No CUDA devices found!");
            exit(0);
        }
        // initialize device
        cudaSetDevice(0);

        printf("Hash Collider - Starting Task 1...\n\n");
        task1();
        return 0;
    #else
        printf("Invalid user input macro evaluation... Please ensure CUDA Version, DeviceQuery Information, and User Variables are correct.\n");
        return -1;
    #endif // VALID_USER_INPUTS
}
