#include "tasks.cuh"


// modified WjCryptLib_Md5 library
typedef struct
{
    uint32_t     lo;
    uint32_t     hi;
    uint32_t     a;
    uint32_t     b;
    uint32_t     c;
    uint32_t     d;
    uint8_t      buffer[64];
    uint32_t     block[16];
} Md5Context_dev;

#define MD5_HASH_SIZE           ( 128 / 8 )

typedef struct
{
    uint8_t      bytes[MD5_HASH_SIZE];
} MD5_HASH_dev;

__device__ void memcpy_dtod(char* dst, char* src, int size)
{
    for (int byte = 0; byte <= size; ++byte)
    {
        dst[byte] = src[byte];
    }
}

__device__ void memset_dtod(char* dst, uint8_t src, int size)
{
    for (int byte = 0; byte <= size; ++byte)
    {
        dst[byte] = src;
    }
}

#define F( x, y, z )            ( (z) ^ ((x) & ((y) ^ (z))) )
#define G( x, y, z )            ( (y) ^ ((z) & ((x) ^ (y))) )
#define H( x, y, z )            ( (x) ^ (y) ^ (z) )
#define I( x, y, z )            ( (y) ^ ((x) | ~(z)) )
#define STEP( f, a, b, c, d, x, t, s )                          \
    (a) += f((b), (c), (d)) + (x) + (t);                        \
    (a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s))));  \
    (a) += (b);

__device__ static void* TransformFunction_dev(Md5Context_dev* ctx, void const* data, uintmax_t size)
{
    uint8_t* ptr;
    uint32_t     a;
    uint32_t     b;
    uint32_t     c;
    uint32_t     d;
    uint32_t     saved_a;
    uint32_t     saved_b;
    uint32_t     saved_c;
    uint32_t     saved_d;

#define GET(n) (ctx->block[(n)])
#define SET(n) (ctx->block[(n)] =             \
            ((uint32_t)ptr[(n)*4 + 0] << 0 )      \
        |   ((uint32_t)ptr[(n)*4 + 1] << 8 )      \
        |   ((uint32_t)ptr[(n)*4 + 2] << 16)      \
        |   ((uint32_t)ptr[(n)*4 + 3] << 24) )

    ptr = (uint8_t*)data;

    a = ctx->a;
    b = ctx->b;
    c = ctx->c;
    d = ctx->d;

    do
    {
        saved_a = a;
        saved_b = b;
        saved_c = c;
        saved_d = d;

        // Round 1
        STEP(F, a, b, c, d, SET(0), 0xd76aa478, 7)
            STEP(F, d, a, b, c, SET(1), 0xe8c7b756, 12)
            STEP(F, c, d, a, b, SET(2), 0x242070db, 17)
            STEP(F, b, c, d, a, SET(3), 0xc1bdceee, 22)
            STEP(F, a, b, c, d, SET(4), 0xf57c0faf, 7)
            STEP(F, d, a, b, c, SET(5), 0x4787c62a, 12)
            STEP(F, c, d, a, b, SET(6), 0xa8304613, 17)
            STEP(F, b, c, d, a, SET(7), 0xfd469501, 22)
            STEP(F, a, b, c, d, SET(8), 0x698098d8, 7)
            STEP(F, d, a, b, c, SET(9), 0x8b44f7af, 12)
            STEP(F, c, d, a, b, SET(10), 0xffff5bb1, 17)
            STEP(F, b, c, d, a, SET(11), 0x895cd7be, 22)
            STEP(F, a, b, c, d, SET(12), 0x6b901122, 7)
            STEP(F, d, a, b, c, SET(13), 0xfd987193, 12)
            STEP(F, c, d, a, b, SET(14), 0xa679438e, 17)
            STEP(F, b, c, d, a, SET(15), 0x49b40821, 22)

            // Round 2
            STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
            STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
            STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
            STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
            STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
            STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
            STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
            STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
            STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
            STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
            STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
            STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
            STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
            STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
            STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
            STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)

            // Round 3
            STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
            STEP(H, d, a, b, c, GET(8), 0x8771f681, 11)
            STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
            STEP(H, b, c, d, a, GET(14), 0xfde5380c, 23)
            STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
            STEP(H, d, a, b, c, GET(4), 0x4bdecfa9, 11)
            STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
            STEP(H, b, c, d, a, GET(10), 0xbebfbc70, 23)
            STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
            STEP(H, d, a, b, c, GET(0), 0xeaa127fa, 11)
            STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
            STEP(H, b, c, d, a, GET(6), 0x04881d05, 23)
            STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
            STEP(H, d, a, b, c, GET(12), 0xe6db99e5, 11)
            STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
            STEP(H, b, c, d, a, GET(2), 0xc4ac5665, 23)

            // Round 4
            STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
            STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
            STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
            STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
            STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
            STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
            STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
            STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
            STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
            STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
            STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
            STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
            STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
            STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
            STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
            STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)

            a += saved_a;
        b += saved_b;
        c += saved_c;
        d += saved_d;

        ptr += 64;
    } while (size -= 64);

    ctx->a = a;
    ctx->b = b;
    ctx->c = c;
    ctx->d = d;

#undef GET
#undef SET

    return ptr;
}

__device__ void Md5Initialise_dev(Md5Context_dev* Context)
{
    Context->a = 0x67452301;
    Context->b = 0xefcdab89;
    Context->c = 0x98badcfe;
    Context->d = 0x10325476;

    Context->lo = 0;
    Context->hi = 0;
}

__device__ void Md5Update_dev(Md5Context_dev* Context, void const* Buffer, uint32_t BufferSize)
{
    uint32_t    saved_lo;
    uint32_t    used;
    uint32_t    free;

    saved_lo = Context->lo;
    if ((Context->lo = (saved_lo + BufferSize) & 0x1fffffff) < saved_lo)
    {
        Context->hi++;
    }
    Context->hi += (uint32_t)(BufferSize >> 29);

    used = saved_lo & 0x3f;

    if (used)
    {
        free = 64 - used;

        if (BufferSize < free)
        {
            memcpy_dtod((char*)&Context->buffer[used], (char*)Buffer, BufferSize);
            return;
        }

        memcpy_dtod((char*)&Context->buffer[used], (char*)Buffer, free);
        Buffer = (uint8_t*)Buffer + free;
        BufferSize -= free;
        TransformFunction_dev(Context, Context->buffer, 64);
    }

    if (BufferSize >= 64)
    {
        Buffer = TransformFunction_dev(Context, Buffer, BufferSize & ~(unsigned long)0x3f);
        BufferSize &= 0x3f;
    }

    memcpy_dtod((char*)Context->buffer, (char*)Buffer, BufferSize);
}

__device__ void Md5Finalise_dev(Md5Context_dev* Context, MD5_HASH_dev* Digest)
{
    uint32_t    used;
    uint32_t    free;

    used = Context->lo & 0x3f;

    Context->buffer[used++] = 0x80;

    free = 64 - used;

    if (free < 8)
    {
        memset_dtod((char*)&Context->buffer[used], 0, free);
        TransformFunction_dev(Context, Context->buffer, 64);
        used = 0;
        free = 64;
    }

    memset_dtod((char*)&Context->buffer[used], 0, free - 8);

    Context->lo <<= 3;
    Context->buffer[56] = (uint8_t)(Context->lo);
    Context->buffer[57] = (uint8_t)(Context->lo >> 8);
    Context->buffer[58] = (uint8_t)(Context->lo >> 16);
    Context->buffer[59] = (uint8_t)(Context->lo >> 24);
    Context->buffer[60] = (uint8_t)(Context->hi);
    Context->buffer[61] = (uint8_t)(Context->hi >> 8);
    Context->buffer[62] = (uint8_t)(Context->hi >> 16);
    Context->buffer[63] = (uint8_t)(Context->hi >> 24);

    TransformFunction_dev(Context, Context->buffer, 64);

    uint8_t test_seg = (uint8_t)(Context->a);
    Digest->bytes[0] = test_seg;
    Digest->bytes[1] = (uint8_t)(Context->a >> 8);
    Digest->bytes[2] = (uint8_t)(Context->a >> 16);
    Digest->bytes[3] = (uint8_t)(Context->a >> 24);
    Digest->bytes[4] = (uint8_t)(Context->b);
    Digest->bytes[5] = (uint8_t)(Context->b >> 8);
    Digest->bytes[6] = (uint8_t)(Context->b >> 16);
    Digest->bytes[7] = (uint8_t)(Context->b >> 24);
    Digest->bytes[8] = (uint8_t)(Context->c);
    Digest->bytes[9] = (uint8_t)(Context->c >> 8);
    Digest->bytes[10] = (uint8_t)(Context->c >> 16);
    Digest->bytes[11] = (uint8_t)(Context->c >> 24);
    Digest->bytes[12] = (uint8_t)(Context->d);
    Digest->bytes[13] = (uint8_t)(Context->d >> 8);
    Digest->bytes[14] = (uint8_t)(Context->d >> 16);
    Digest->bytes[15] = (uint8_t)(Context->d >> 24);
}

__device__ void Md5Calculate_dev(void  const* Buffer, uint32_t BufferSize, MD5_HASH_dev* Digest)
{
    Md5Context_dev context;

    Md5Initialise_dev(&context);
    Md5Update_dev(&context, Buffer, BufferSize);
    Md5Finalise_dev(&context, Digest);
}




// statically initialized global variables
__device__ volatile uint8_t d_num_collisions_found = 0;              // track number of collisions found by active kernel
__device__ volatile unsigned long long d_collision_attempts = 0;     // track approximate cumulative number of attempts
__device__ volatile int d_collision_flag = FALSE;                    // signal host to read
__device__ volatile int global_mutex = UNLOCKED;                     // signal mutex to other threads (globally)
__device__ volatile unsigned long long global_seeder = 0;            // increment before each rand call to avoid repeat deterministic "randoms"


// dynamically initialized in host
__constant__ __device__ volatile MD5_HASH_dev d_const_md5_digest;    // store digest on L2 or L1 cache (on v8.6)
__device__ volatile unsigned long long d_collision_size;             // track # of characters in collision

__global__ void find_collisions(volatile char* collision) {
    //===========================================================================================================
    // DECLARATIONS & INITIALIZATION
    //===========================================================================================================
    printf("\n...Entering thread %d.\n", threadIdx.x);

    // create warp synchronization variables
    int sync_warp_flag = FALSE;
    int lane_id_of_found_collision = -1;

    // allocate local buffer todo make this type a macro
    char local_collision[INITIAL_COLLISION_BUFF_SIZE];

    // initialize local buffer
    unsigned long long local_collision_size = d_collision_size;
    for (int byte_index = 0; byte_index <= local_collision_size; ++byte_index) {
        local_collision[byte_index] = collision[byte_index];
    }
    printf("Initialized local collision storage: %s.\n", local_collision);
    printf("Local storage currently using %llu characters.\n", local_collision_size);

    // allocate room for new digest
    MD5_HASH_dev local_md5_digest;

    // allocate storage for random character
    RAND_TYPE randoms[NUM_RANDOMS];
    unsigned long long random_index = NUM_RANDOMS;
    printf("Initialized randoms.\n");

    //===========================================================================================================
    // COMPUTATIONS - GENERATE RANDS, RESIZE BUFFER, APPEND CHAR, HASH, COMPARE { EXIT }
    //===========================================================================================================

    while (d_num_collisions_found < TARGET_COLLISIONS)
    {
        // generate a new batch of random numbers and append a random char to collision
        if (random_index == NUM_RANDOMS)
        {
            // generate a new batch of randoms todo: make each random unique to the set
            random_index = 0;
            for (int i = 0; i < FUNCTION_CALLS_NEEDED; ++i) {
                curandStatePhilox4_32_10_t state;
                int tid = threadIdx.x;  // randomize w/in warps (w/in lockstep)
                ++global_seeder;        // randomize between warps (and blocks)
                curand_init(tid+global_seeder, i, 0, &state);
                // effectively assigns 4 random uint8_t's per execution
                *((uint32_t *) (randoms + i * RAND_BIT_MULTIPLE)) = curand(&state);
            }

            /*
            // resize local_collision
            if (local_collision_size == local_collision_size) {
                // retain ptr to old buffer
                char* old_buff = local_collision;

                // reassign local_collision ptr to new buffer
                local_buff_size *= 2;
                cudaMalloc(&local_collision, local_buff_size);

                // copy data from old buffer to new buffer
                for (int i = 0; i < ARBITRARY_MAX_BUFF_SIZE; ++i) {
                    local_collision[i] = old_buff[i];
                }

                // free original buffer
                //cudaFree(old_buff);
            }
            */

            // permanently append a random char from the final position in randoms[]
            ++local_collision_size;
        }

        // try (1..2^NUM_ENCODING_BITS-1) for local collision index

        for (int c = 1; c < CHAR_SET_SIZE; ++c) {
            // try charset
            local_collision[local_collision_size - 1] = c;
            local_collision[local_collision_size] = '\0';

            // generate new hash
            Md5Calculate_dev((const void *) local_collision, local_collision_size, &local_md5_digest);
            for (int i = 0; i < MD5_HASH_SIZE / 2; ++i) {
                printf("%2.2x", local_md5_digest.bytes[i]);
            }
            printf(" | ");
            for (int i = 0; i < MD5_HASH_SIZE / 2; ++i) {
                printf("%2.2x", d_const_md5_digest.bytes[i]);
            }
            printf(" | Attempt: %llu\n", d_collision_attempts);
            ++d_collision_attempts;

            // we found a collision!
            if ((*((uint32_t *) &d_const_md5_digest) >> 12 == *((uint32_t *) &local_md5_digest) >> 12) || sync_warp_flag == TRUE) {
                printf("%d thread id found hash first.\n", threadIdx.x);

                // synchronize all threads in warp, and get "value" from lane 0
                if (lane_id_of_found_collision == -1) {
                    lane_id_of_found_collision = threadIdx.x & (WARP_SIZE - 1);
                    __shfl_sync(0xffffffff, lane_id_of_found_collision, lane_id_of_found_collision);
                }

                // synchronize all threads in warp, and get "value" from lane 0
                sync_warp_flag = TRUE;
                __shfl_sync(0xffffffff, sync_warp_flag, lane_id_of_found_collision);

                // threads will converge so that critical thread is not indefinitely suspended
                while (sync_warp_flag) {
                    // do mutex operations
                    if ((threadIdx.x & 0x1f) == lane_id_of_found_collision) {
                        printf("Thread %d in lane %d entered warp mutex.\n", threadIdx.x, lane_id_of_found_collision);
                        // set mutex lock
                        do {} while (atomicCAS((int*)&global_mutex, UNLOCKED, LOCKED));

                        // enter critical section - writing for host polls and signalling when ready
                        for (int byte_index = 0; byte_index <= local_collision_size; ++byte_index) {
                            collision[byte_index] = local_collision[byte_index];
                        }
                        d_collision_size = local_collision_size;

                        // signal host to read
                        d_collision_flag = TRUE;

                        // free lock only once host signals finished reading (e.g. d_collision_flag = FALSE)
                        do {
                            // thread idles while host updates
                            // 1) d_num_collisions_found
                            // 2) d_collision_flag
                        } while (d_collision_flag);

                        // safely unlock mutex by writing to flag - remember relaxed ordering doesn't matter here
                        atomicExch((int*)&global_mutex, UNLOCKED);

                        // release non-critical warp threads and reset flag
                        sync_warp_flag = FALSE;
                        __shfl_sync(0xffffffff, sync_warp_flag, lane_id_of_found_collision);
                    } else {
                        // have non-critical warp threads read check for
                    }
                    __syncwarp(); // causes the executing thread to wait until all threads specified in mask have executed a __syncwarp()
                }
            }
        }

        // on next iter: try next random || gen new randoms and append last random to local_collision
        local_collision[local_collision_size - 1] = randoms[random_index];
        local_collision[local_collision_size] = '\0';
        ++random_index;
    }
}

void task1() {
    //===========================================================================================================
    // SEQUENTIAL TASKS (Initial)
    //===========================================================================================================

    // todo v5 cudaMallocHost - code chunk has been tested
    // char* h_page_locked_data;
    // gpuErrchk( cudaMallocHost(&h_page_locked_data, ARBITRARY_MAX_BUFF_SIZE) );
    // cudaMemset(&h_page_locked_data, 0x00, sizeof(char) * ARBITRARY_MAX_BUFF_SIZE);

    // read file data
    char sampleFile_path[] = "/home/shford/CLionProjects/cuda_hashing/sample.txt";
    char* h_sampleFile_buff;
    unsigned long long h_sampleFile_buff_size = 0; // handle files up to ~4GiB (2^32-1 bytes) -- may be 1 byte too small
    get_file_data((char*)sampleFile_path, &h_sampleFile_buff, (uint32_t*)&h_sampleFile_buff_size);

    // get hash md5_digest
    MD5_HASH md5_digest;
    Md5Calculate((const void*)h_sampleFile_buff, h_sampleFile_buff_size, &md5_digest);

    // format and print digest as a string of hex characters
    char hash[MD5_HASH_SIZE_B + 1]; //MD5 len is 16B, 1B = 2 chars

    char tiny_hash[TINY_HASH_SIZE_B + 1];
    for (int i = 0; i < MD5_HASH_SIZE / 2; ++i)
    {
        sprintf(hash + i * 2, "%2.2x", md5_digest.bytes[i]);
    }
    hash[sizeof(hash)-1] = '\0';
    strncpy(tiny_hash, hash, sizeof(tiny_hash));
    tiny_hash[sizeof(tiny_hash)-1] = '\0';

    printf("Full MD5 md5_digest is: %s\n", hash);
    printf("TinyHash md5_digest is: %s\n\n", tiny_hash);

    //===========================================================================================================
    // BEGIN CUDA PARALLELIZATION
    //===========================================================================================================

    // allocate storage for collisions once found
    uint8_t h_collision_index = 0;
    char* h_collisions[TARGET_COLLISIONS];
    unsigned long long h_collision_sizes[TARGET_COLLISIONS];
    unsigned long long h_collision_attempts = 0;
    for (int i = 0; i < TARGET_COLLISIONS; ++i) {
        h_collisions[i] = (char*)calloc(1, INITIAL_COLLISION_BUFF_SIZE);
        h_collision_sizes[i] = 0;
    }
    int h_collision_flag = FALSE;
    printf("...Allocated host variables.\n");

    // allocate global mem for collision - initialized in loop
    volatile char* d_collision;
    gpuErrchk( cudaMalloc((void **)&d_collision, INITIAL_COLLISION_BUFF_SIZE) );
    gpuErrchk( cudaMemset((void*)d_collision, 0x00, INITIAL_COLLISION_BUFF_SIZE));
    printf("...Allocated device collision.\n");

    // parallelization setup - initialize device globals
    gpuErrchk( cudaMemcpyToSymbol(d_const_md5_digest, &md5_digest, sizeof(md5_digest), 0, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpyToSymbol(d_collision_size, &h_sampleFile_buff_size, sizeof(h_sampleFile_buff_size), 0, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy((void*)d_collision, h_sampleFile_buff, h_sampleFile_buff_size, cudaMemcpyHostToDevice) );
    printf("...Initialized dynamic device variables.\n");

    // execution configuration (sync device)
    printf("...Launching kernel.\n");
    find_collisions<<<BLOCKS_PER_KERNEL, THREADS_PER_BLOCK>>>(d_collision);
    printf("Done launching kernel.\n");

    // poll kernel
    while (h_collision_index < TARGET_COLLISIONS)
    {
        // poll collision flag
        while (!h_collision_flag)
        {
            // NOTE: errors on this line are most likely from the kernel
            gpuErrchk( cudaMemcpyFromSymbol(&h_collision_flag, d_collision_flag, sizeof(h_collision_flag), 0, cudaMemcpyDeviceToHost));
        }

        // read updated collision size, collision
        printf("...Reading collision.\n");
        gpuErrchk( cudaMemcpyFromSymbol(&h_collision_sizes[h_collision_index], d_collision_size, sizeof(h_collision_sizes[h_collision_index]), 0, cudaMemcpyDeviceToHost));
        gpuErrchk( cudaMemcpy(&h_collisions[h_collision_index], (const void*)d_collision, h_collision_sizes[h_collision_index], cudaMemcpyDeviceToHost) );
        printf("Done.\n\n");

        // increment collision index for host and device
        printf("...Incrementing collision counter.\n");
        ++h_collision_index;
        gpuErrchk( cudaMemcpyToSymbol(d_num_collisions_found, &h_collision_index, sizeof(h_collision_index), 0, cudaMemcpyHostToDevice));
        printf("Done.\n\n");

        // reset flags to release device-wide mutex and reset kernel
        printf("...Resetting device flag.\n");
        h_collision_flag = FALSE;
        cudaMemcpyToSymbol(d_collision_flag, &h_collision_flag, sizeof(h_collision_flag), 0, cudaMemcpyHostToDevice);
        printf("Done.\n\n");
    }
    
    // read the final number of cumulative hashing attempts
    if (h_collision_index == TARGET_COLLISIONS)
    {
        gpuErrchk(cudaMemcpyFromSymbol(&h_collision_attempts, d_collision_attempts,
            sizeof(h_collision_attempts), 0, cudaMemcpyDeviceToHost));
    }

    printf("...Synchronizing threads.\n");
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Done.\n\n");
    printf("...Resetting device.\n");
    gpuErrchk( cudaDeviceReset() );
    printf("Done.\n\n");

    // free collisions
    printf("...Freeing memory.\n");
    cudaFree((void*)d_collision);
    free(h_sampleFile_buff);
    printf("Done.\n\n");

    printf("\nCalculated %d collisions in %lld attempts... Success!/\n", TARGET_COLLISIONS, h_collision_attempts);

    //===========================================================================================================
    // WRITE COLLISIONS TO DISK
    //===========================================================================================================

    printf("Original string: %s\n", h_sampleFile_buff);
    for (int i = 0; i < TARGET_COLLISIONS; ++i) {
        printf("Collision %d: %s\n", i, h_collisions[i]);

        // todo write collision

        // free collision once written
        free(h_collisions[i]);
    }
}
