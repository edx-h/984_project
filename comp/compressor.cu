#include <cstddef>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <cuda_runtime.h>
#include <chrono>

#define CHUNK_SIZE 1024
#define MAX_COMPRESSED_CHUNK_SIZE (CHUNK_SIZE)
#define DICT_SIZE 16

__constant__ unsigned char d_dictionary[DICT_SIZE] = { ' ', 'e', 't', 'a', 'o', 'i', 'n', 's', 'r', 'h', 'l', 'd', 'u', 'c', 'm', 'f' };

// **Compression Kernel with Coalesced Memory Access**
__global__ void compressKernelBitPack_coalesced(const unsigned char *d_input, int input_size, int chunk_size, 
                                                unsigned char *d_tempOut, int *d_chunkBitSizes)
{
    int chunk_id = blockIdx.x;
    int chunk_start = chunk_id * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > input_size) chunk_end = input_size;
    int n = chunk_end - chunk_start;

    extern __shared__ int s_bitLens[];

    // Coalesced access: consecutive threads read consecutive memory locations
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        unsigned char symbol = d_input[chunk_start + i];
        bool found = false;
        int dict_index = 0;
        for (int d = 0; d < DICT_SIZE; d++) {
            if (symbol == d_dictionary[d]) {
                found = true;
                dict_index = d;
                break;
            }
        }
        s_bitLens[i] = found ? 5 : 9;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int totalBits = 0;
        for (int i = 0; i < n; i++) {
            int tmp = s_bitLens[i];
            s_bitLens[i] = totalBits;
            totalBits += tmp;
        }
        d_chunkBitSizes[chunk_id] = totalBits;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned char *out = d_tempOut + chunk_id * MAX_COMPRESSED_CHUNK_SIZE;
        for (int i = 0; i < MAX_COMPRESSED_CHUNK_SIZE; i++) out[i] = 0;
        int bitPos = 0;
        for (int i = 0; i < n; i++) {
            unsigned char symbol = d_input[chunk_start + i];
            bool found = false;
            int dict_index = 0;
            for (int d = 0; d < DICT_SIZE; d++) {
                if (symbol == d_dictionary[d]) {
                    found = true;
                    dict_index = d;
                    break;
                }
            }
            int code = found ? ((1 << 4) | (dict_index & 0xF)) : symbol;
            int bits = found ? 5 : 9;
            int remaining = bits;
            while (remaining > 0) {
                int bytePos = bitPos / 8;
                int bitOffset = bitPos % 8;
                int space = 8 - bitOffset;
                int writeBits = (remaining < space) ? remaining : space;
                int shift = remaining - writeBits;
                int bitsToWrite = (code >> shift) & ((1 << writeBits) - 1);
                out[bytePos] |= bitsToWrite << (space - writeBits);
                bitPos += writeBits;
                remaining -= writeBits;
            }
        }
    }
}

__global__ void compressKernelBitPack_noncoalesced(const unsigned char *d_input, int input_size, int chunk_size, 
                                                   unsigned char *d_tempOut, int *d_chunkBitSizes)
{
    int chunk_id = blockIdx.x;
    int chunk_start = chunk_id * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > input_size) chunk_end = input_size;
    int n = chunk_end - chunk_start;

    extern __shared__ int s_bitLens[];

    // 非合并访存：使用质数步长打乱访问顺序
    int total_threads = blockDim.x;
    int tid = threadIdx.x;
    int start_idx = (tid * 17) % total_threads; // 17是质数，确保覆盖所有符号
    for (int i = start_idx; i < n; i += total_threads) {
        unsigned char symbol = d_input[chunk_start + i];
        bool found = false;
        int dict_index = 0;
        for (int d = 0; d < DICT_SIZE; d++) {
            if (symbol == d_dictionary[d]) {
                found = true;
                dict_index = d;
                break;
            }
        }
        s_bitLens[i] = found ? 5 : 9; // 记录符号的bit长度
    }
    __syncthreads();

    // 线程0计算bit长度前缀和
    if (threadIdx.x == 0) {
        int totalBits = 0;
        for (int i = 0; i < n; i++) {
            int tmp = s_bitLens[i];
            s_bitLens[i] = totalBits;
            totalBits += tmp;
        }
        d_chunkBitSizes[chunk_id] = totalBits;
    }
    __syncthreads();

    // 线程0执行bit-packing
    if (threadIdx.x == 0) {
        unsigned char *out = d_tempOut + chunk_id * MAX_COMPRESSED_CHUNK_SIZE;
        for (int i = 0; i < MAX_COMPRESSED_CHUNK_SIZE; i++) out[i] = 0; // 清零输出缓冲区
        int bitPos = 0;
        for (int i = 0; i < n; i++) {
            unsigned char symbol = d_input[chunk_start + i];
            bool found = false;
            int dict_index = 0;
            for (int d = 0; d < DICT_SIZE; d++) {
                if (symbol == d_dictionary[d]) {
                    found = true;
                    dict_index = d;
                    break;
                }
            }
            int code = found ? ((1 << 4) | (dict_index & 0xF)) : symbol;
            int bits = found ? 5 : 9;
            int remaining = bits;
            while (remaining > 0) {
                int bytePos = bitPos / 8;
                int bitOffset = bitPos % 8;
                int space = 8 - bitOffset;
                int writeBits = (remaining < space) ? remaining : space;
                int shift = remaining - writeBits;
                int bitsToWrite = (code >> shift) & ((1 << writeBits) - 1);
                out[bytePos] |= bitsToWrite << (space - writeBits);
                bitPos += writeBits;
                remaining -= writeBits;
            }
        }
    }
}

// **Compute Offsets Kernel**
__global__ void computeOffsetsKernel(const int *d_chunkBitSizes, int *d_chunkByteOffsets, int num_chunks)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int sum = 0;
        d_chunkByteOffsets[0] = 0;
        for (int i = 0; i < num_chunks; i++) {
            int byteSize = (d_chunkBitSizes[i] + 7) / 8;
            sum += byteSize;
            d_chunkByteOffsets[i + 1] = sum;
        }
    }
}

// **Parallel Assembly Kernel**
__global__ void assembleKernelBitPack_parallel(const unsigned char *d_tempOut, const int *d_chunkByteOffsets, int num_chunks, unsigned char *d_finalOut)
{
    int chunk_id = blockIdx.x;
    if (chunk_id >= num_chunks) return;
    int start = d_chunkByteOffsets[chunk_id];
    int end = d_chunkByteOffsets[chunk_id + 1];
    int size = end - start;
    const unsigned char *src = d_tempOut + chunk_id * MAX_COMPRESSED_CHUNK_SIZE;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        d_finalOut[start + i] = src[i];
    }
}

// **Serial Assembly Kernel**
__global__ void assembleKernelBitPack_serial(const unsigned char *d_tempOut, const int *d_chunkByteOffsets, int num_chunks, unsigned char *d_finalOut)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
            int start = d_chunkByteOffsets[chunk_id];
            int end = d_chunkByteOffsets[chunk_id + 1];
            int size = end - start;
            const unsigned char *src = d_tempOut + chunk_id * MAX_COMPRESSED_CHUNK_SIZE;
            for (int i = 0; i < size; i++) {
                d_finalOut[start + i] = src[i];
            }
        }
    }
}

// **Decompression Kernel**
__global__ void decompressKernelBitPack(const unsigned char *d_compressed, const int *d_chunkBitSizes,
                                        const int *d_chunkByteOffsets, int chunk_size, 
                                        unsigned char *d_output, int input_size,
                                        int *d_decompSize)
{
    int chunk_id = blockIdx.x;
    int chunk_start = chunk_id * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > input_size) chunk_end = input_size;
    int n = chunk_end - chunk_start;

    __shared__ int totalBits;
    if (threadIdx.x == 0) totalBits = d_chunkBitSizes[chunk_id];
    __syncthreads();

    int offset = d_chunkByteOffsets[chunk_id];
    const unsigned char *in = d_compressed + offset;
    int bitPos = 0;

    for (int i = 0; i < n; i++) {
        if (bitPos >= totalBits) break;
        int bytePos = bitPos / 8;
        int bitOffset = bitPos % 8;
        unsigned char curByte = in[bytePos];
        int flag = (curByte >> (7 - bitOffset)) & 0x1;
        bitPos += 1;
        unsigned char decoded = 0;
        if (flag == 1) {
            int index = 0;
            int remaining = 4;
            while (remaining > 0) {
                bytePos = bitPos / 8;
                bitOffset = bitPos % 8;
                int space = 8 - bitOffset;
                int readBits = (remaining < space) ? remaining : space;
                int shift = space - readBits;
                int bitsRead = (in[bytePos] >> shift) & ((1 << readBits) - 1);
                index = (index << readBits) | bitsRead;
                bitPos += readBits;
                remaining -= readBits;
            }
            decoded = d_dictionary[index];
        } else {
            int val = 0;
            int remaining = 8;
            while (remaining > 0) {
                bytePos = bitPos / 8;
                bitOffset = bitPos % 8;
                int space = 8 - bitOffset;
                int readBits = (remaining < space) ? remaining : space;
                int shift = space - readBits;
                int bitsRead = (in[bytePos] >> shift) & ((1 << readBits) - 1);
                val = (val << readBits) | bitsRead;
                bitPos += readBits;
                remaining -= readBits;
            }
            decoded = (unsigned char)val;
        }
        d_output[chunk_start + i] = decoded;
    }

    if (threadIdx.x == 0) atomicAdd(d_decompSize, n);
}

// **Read File Function**
unsigned char* readFile(const char* filePath, int* fileSize) {
    FILE *file = fopen(filePath, "rb");
    if (!file) {
        printf("无法打开文件: %s\n", filePath);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    *fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    unsigned char *buffer = (unsigned char*)malloc(*fileSize);
    fread(buffer, sizeof(unsigned char), *fileSize, file);
    fclose(file);
    return buffer;
}

// **Process File Function**
void processFile(const char* filePath, int switchValue,long long& total_duration, 
                 long long& total_duration_h2d, 
                 long long& total_duration_gpu_compress, 
                 long long& total_duration_compute_offset, 
                 long long& total_duration_assemble, 
                 long long& total_duration_d2h, 
                 long long& total_duration_decompress,
                size_t& file_count) {
    int input_size;
    unsigned char* testStr = readFile(filePath, &input_size);
    if (!testStr) return;

    printf("正在处理文件: %s (大小: %d bytes)\n", filePath, input_size);

    auto start_h2d = std::chrono::high_resolution_clock::now();
    unsigned char *d_input;
    cudaMalloc((void**)&d_input, input_size * sizeof(unsigned char));
    cudaMemcpy(d_input, testStr, input_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int num_chunks = (input_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

    unsigned char *d_tempOut;
    cudaMalloc((void**)&d_tempOut, num_chunks * MAX_COMPRESSED_CHUNK_SIZE * sizeof(unsigned char));

    int *d_chunkBitSizes;
    cudaMalloc((void**)&d_chunkBitSizes, num_chunks * sizeof(int));

    auto end_h2d = std::chrono::high_resolution_clock::now();

    auto start_gpu_compress = std::chrono::high_resolution_clock::now();
    int sharedMemSize = CHUNK_SIZE * sizeof(int);

    // Select compression kernel based on switch value
    if (switchValue == 0 || switchValue == 2) {
        compressKernelBitPack_coalesced<<<num_chunks, 256, sharedMemSize>>>(d_input, input_size, CHUNK_SIZE, d_tempOut, d_chunkBitSizes);
    } else {
        compressKernelBitPack_noncoalesced<<<num_chunks, 256, sharedMemSize>>>(d_input, input_size, CHUNK_SIZE, d_tempOut, d_chunkBitSizes);
    }
    cudaDeviceSynchronize();
    auto end_gpu_compress = std::chrono::high_resolution_clock::now();

    auto start_compute_offset = std::chrono::high_resolution_clock::now();
    int *d_chunkByteOffsets;
    cudaMalloc((void**)&d_chunkByteOffsets, (num_chunks + 1) * sizeof(int));
    computeOffsetsKernel<<<1, 1>>>(d_chunkBitSizes, d_chunkByteOffsets, num_chunks);
    cudaDeviceSynchronize();
    auto end_compute_offset = std::chrono::high_resolution_clock::now();

    auto start_assemble = std::chrono::high_resolution_clock::now();
    int h_finalSize;
    cudaMemcpy(&h_finalSize, d_chunkByteOffsets + num_chunks, sizeof(int), cudaMemcpyDeviceToHost);
    printf("压缩后总字节数: %d bytes\n", h_finalSize);

    unsigned char *d_finalOut;
    cudaMalloc((void**)&d_finalOut, h_finalSize * sizeof(unsigned char));

    // Select assembly kernel based on switch value
    if (switchValue == 0 || switchValue == 1) {
        assembleKernelBitPack_parallel<<<num_chunks, 256>>>(d_tempOut, d_chunkByteOffsets, num_chunks, d_finalOut);
    } else {
        assembleKernelBitPack_serial<<<1, 1>>>(d_tempOut, d_chunkByteOffsets, num_chunks, d_finalOut);
    }
    cudaDeviceSynchronize();
    auto end_assemble = std::chrono::high_resolution_clock::now();

    auto start_d2h = std::chrono::high_resolution_clock::now();
    unsigned char *h_finalOut = (unsigned char*)malloc(h_finalSize * sizeof(unsigned char));
    cudaMemcpy(h_finalOut, d_finalOut, h_finalSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    auto end_compress = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_compress - start_h2d).count();
    auto duration_h2d = std::chrono::duration_cast<std::chrono::microseconds>(end_h2d-start_h2d).count();
    auto duration_gpu_compress = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_compress-start_gpu_compress).count();
    auto duration_compute_offset = std::chrono::duration_cast<std::chrono::microseconds>(end_compute_offset-start_compute_offset).count();
    auto duration_assemble = std::chrono::duration_cast<std::chrono::microseconds>(end_assemble-start_assemble).count();
    auto duration_d2h = std::chrono::duration_cast<std::chrono::microseconds>(end_compress-start_d2h).count();

    // 累加到全局变量
    total_duration += duration;
    total_duration_h2d += duration_h2d;
    total_duration_gpu_compress += duration_gpu_compress;
    total_duration_compute_offset += duration_compute_offset;
    total_duration_assemble += duration_assemble;
    total_duration_d2h += duration_d2h;

    FILE *file = fopen("compressed_output.bin", "wb");
    if (file) {
        fwrite(h_finalOut, sizeof(unsigned char), h_finalSize, file);
        fclose(file);
        printf("压缩数据已成功写入文件 compressed_output.bin\n");
    } else {
        printf("无法打开文件进行写入\n");
    }

    auto start_decomp = std::chrono::high_resolution_clock::now();

    unsigned char *d_decompOut;
    cudaMalloc((void**)&d_decompOut, input_size * sizeof(unsigned char));

    int *d_decompSize;
    cudaMalloc((void**)&d_decompSize, sizeof(int));
    cudaMemset(d_decompSize, 0, sizeof(int));

    decompressKernelBitPack<<<num_chunks, 256, sizeof(int)>>>(d_finalOut, d_chunkBitSizes, d_chunkByteOffsets, CHUNK_SIZE, d_decompOut, input_size, d_decompSize);
    cudaDeviceSynchronize();

    unsigned char *h_decompOut = (unsigned char*)malloc(input_size * sizeof(unsigned char));
    cudaMemcpy(h_decompOut, d_decompOut, input_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    int h_decompSize;
    cudaMemcpy(&h_decompSize, d_decompSize, sizeof(int), cudaMemcpyDeviceToHost);
    printf("解压后的数据字节数: %d bytes\n", h_decompSize);

    auto end_decomp = std::chrono::high_resolution_clock::now();
    auto duration_decompress = std::chrono::duration_cast<std::chrono::microseconds>(end_decomp - start_decomp).count();
    printf("解压缩耗时: %ld μs\n", duration_decompress);
    total_duration_decompress += duration_decompress;

    free(h_finalOut);
    free(h_decompOut);
    free(testStr);
    cudaFree(d_input);
    cudaFree(d_tempOut);
    cudaFree(d_chunkBitSizes);
    cudaFree(d_chunkByteOffsets);
    cudaFree(d_finalOut);
    cudaFree(d_decompOut);
    cudaFree(d_decompSize);
    file_count += 1;
}

// **Main Function**
int main(int argc, char **argv) {
    if (argc != 3) {
        printf("用法: %s <目录路径> <开关值>\n开关值: 0=全开, 1=关闭合并访存, 2=关闭并行组装, 3=全关\n", argv[0]);
        return 1;
    }
    const char *directoryPath = argv[1];
    int switchValue = atoi(argv[2]);
    if (switchValue < 0 || switchValue > 3) {
        printf("开关值必须在0到3之间\n");
        return 1;
    }

    // 定义累加变量
    long long total_duration = 0;
    long long total_duration_h2d = 0;
    long long total_duration_gpu_compress = 0;
    long long total_duration_compute_offset = 0;
    long long total_duration_assemble = 0;
    long long total_duration_d2h = 0;
    long long total_duration_decompress = 0;
    int counter{0};
    
    while (counter < 10)
    {
        DIR *dir = opendir(directoryPath);
        if (!dir) {
            printf("无法打开目录: %s\n", directoryPath);
            return 1;
        }
        size_t file_count = 0;
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_name[0] == '.') continue;
            char filePath[512];
            snprintf(filePath, sizeof(filePath), "%s/%s", directoryPath, entry->d_name);
            processFile(filePath, switchValue, total_duration, total_duration_h2d, total_duration_gpu_compress, 
                        total_duration_compute_offset, total_duration_assemble, total_duration_d2h, total_duration_decompress, file_count);
        }
        printf("处理文件数：%zu个\n", file_count);
        ++counter;
        closedir(dir);
    }
    // 打印累计耗时
    printf("\n所有文件处理完毕\n");
    printf("累计压缩总耗时: %lld ms\n", total_duration/10000);
    printf("\t累计H2D: %lld ms\n", total_duration_h2d/10000);
    printf("\t累计GPU压缩: %lld ms\n", total_duration_gpu_compress/10000);
    printf("\t累计计算offset: %lld ms\n", total_duration_compute_offset/10000);
    printf("\t累计组装: %lld ms\n", total_duration_assemble/10000);
    printf("\t累计D2H: %lld ms\n", total_duration_d2h/10000);
    printf("累计解压缩耗时: %lld ms\n", total_duration_decompress/10000);
    return 0;
}