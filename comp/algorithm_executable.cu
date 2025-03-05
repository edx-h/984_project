#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <cuda_runtime.h>
#include <chrono>

#define CHUNK_SIZE 1024   // 每个块的原始数据大小（字节）
// 为简化，假定压缩后每块数据不会超过原始大小（实际中可能更小）
#define MAX_COMPRESSED_CHUNK_SIZE (CHUNK_SIZE)
#define DICT_SIZE 16      // 字典条目数

// 将字典存放在设备常量内存中
// 本例中即使每个字符本来占1字节，由于只需4位表示字典索引，再加1位标记（5位）即可压缩
__constant__ unsigned char d_dictionary[DICT_SIZE] = { ' ', 'e', 't', 'a', 'o', 'i', 'n', 's', 'r', 'h', 'l', 'd', 'u', 'c', 'm', 'f' };

//=====================================================================
// compressKernelBitPack：每个block处理一个输入块，完成以下工作：
// ① 每个线程并行判断自己负责的符号是否命中字典，并记录编码所需的bit数（命中：5位；不命中：9位）。
// ② 利用共享内存存放每个符号编码所需的bit数，由线程0串行计算前缀和（即各符号在输出中的bit偏移），
//    同时累加得到本块总bit数，并写入全局数组 d_chunkBitSizes。
// ③ 线程0依次遍历本块所有符号，根据编码规则进行 bit-packing，将结果写入预分配的临时输出区域。
//=====================================================================
__global__ void compressKernelBitPack(const unsigned char *d_input, int input_size, int chunk_size, 
                                        unsigned char *d_tempOut, int *d_chunkBitSizes)
{
    int chunk_id = blockIdx.x;             // 当前块编号
    int chunk_start = chunk_id * chunk_size; // 本块在全局输入数据中的起始位置
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > input_size) chunk_end = input_size;
    int n = chunk_end - chunk_start;       // 本块实际字符数

    // 分配共享内存，用于存放每个符号编码所需的bit数
    extern __shared__ int s_bitLens[];

    // 每个线程处理部分符号，判断是否命中字典
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

    // 线程0计算前缀和，并得到本块总bit数
    int totalBits = 0;
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            int tmp = s_bitLens[i];
            s_bitLens[i] = totalBits; // 保存该符号的bit起始偏移
            totalBits += tmp;
        }
        d_chunkBitSizes[chunk_id] = totalBits;
    }
    __syncthreads();

    // 线程0进行 bit-packing，将本块压缩数据写入预分配的临时输出区域
    if (threadIdx.x == 0) {
        unsigned char *out = d_tempOut + chunk_id * MAX_COMPRESSED_CHUNK_SIZE;
        for (int i = 0; i < MAX_COMPRESSED_CHUNK_SIZE; i++) {
            out[i] = 0;
        }
        int bitPos = 0;  // 当前写入位置（以bit为单位）
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
            int code = 0;   // 最终编码
            int bits = 0;   // 该字符编码占用的bit数
            if (found) {
                // 命中：5位编码，最高位为1，其余4位存放字典索引
                code = (1 << 4) | (dict_index & 0xF);
                bits = 5;
            } else {
                // 未命中：9位编码，最高位为0，后续8位存原字符
                code = symbol;
                bits = 9;
            }
            // 将 code 中的 bits 位写入输出缓冲区
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

//=====================================================================
// computeOffsetsKernel：单线程kernel，在GPU上计算所有压缩数据块的字节偏移。
// 输入：d_chunkBitSizes，每块压缩数据的bit数（由compressKernelBitPack计算得到）；
// 输出：d_chunkByteOffsets，每块在最终输出中的字节偏移（长度为 num_chunks+1），
 // 其中 d_chunkByteOffsets[0] = 0, d_chunkByteOffsets[i+1] = d_chunkByteOffsets[i] + ceil(bit数/8)。
//=====================================================================
__global__ void computeOffsetsKernel(const int *d_chunkBitSizes, int *d_chunkByteOffsets, int num_chunks)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int sum = 0;
        d_chunkByteOffsets[0] = 0;
        for (int i = 0; i < num_chunks; i++) {
            int byteSize = (d_chunkBitSizes[i] + 7) / 8;
            sum += byteSize;
            d_chunkByteOffsets[i+1] = sum;
        }
    }
}

//=====================================================================
// assembleKernelBitPack：将各块压缩数据组装到最终连续输出缓冲区。
//=====================================================================
__global__ void assembleKernelBitPack(const unsigned char *d_tempOut, const int *d_chunkByteOffsets, int num_chunks, unsigned char *d_finalOut)
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

//=====================================================================
// decompressKernelBitPack：每个block解压一个压缩数据块，采用bit-level解析恢复原始数据。
// 解压规则：
//   1. 读取1位标记；若标记为1，则再读取4位作为字典索引，从字典中取出字符；
//   2. 若标记为0，则读取后续8位作为原始字符。
// 注意：参数顺序已修改为：
//   (d_compressed, d_chunkBitSizes, d_chunkByteOffsets, chunk_size, d_output, input_size)
// 这样调用时第三个参数传入的是偏移数组，而第四个参数传入的是每块大小（int）。
//=====================================================================
__global__ void decompressKernelBitPack(const unsigned char *d_compressed, const int *d_chunkBitSizes,
                                          const int *d_chunkByteOffsets, int chunk_size, 
                                          unsigned char *d_output, int input_size,
                                          int *d_decompSize)
{
    int chunk_id = blockIdx.x;
    int chunk_start = chunk_id * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > input_size) chunk_end = input_size;
    int n = chunk_end - chunk_start;  // 本块实际解压出的字节数

    __shared__ int totalBits;
    if (threadIdx.x == 0) {
        totalBits = d_chunkBitSizes[chunk_id];
    }
    __syncthreads();

    // 从预先计算的偏移位置开始读取该块压缩数据
    int offset = d_chunkByteOffsets[chunk_id];
    const unsigned char *in = d_compressed + offset;
    int bitPos = 0;

    // 逐字符解码
    for (int i = 0; i < n; i++) {
        if (bitPos >= totalBits) break;
        int bytePos = bitPos / 8;
        int bitOffset = bitPos % 8;
        unsigned char curByte = in[bytePos];
        int flag = (curByte >> (7 - bitOffset)) & 0x1;
        bitPos += 1;  // 消耗1位标记
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
    
    // 将本块解压出的字节数累加到全局计数器
    if (threadIdx.x == 0) {
        atomicAdd(d_decompSize, n);
    }
}


// 读取文件内容
unsigned char* readFile(const char* filePath, int* fileSize) {
    FILE *file = fopen(filePath, "rb");
    if (file == NULL) {
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

//=====================================================================
// processFile：处理单个文件的压缩与解压
//=====================================================================
void processFile(const char* filePath){
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    int input_size;
    unsigned char* testStr = readFile(filePath, &input_size);
    if (!testStr) return;

    printf("正在处理文件: %s (大小: %d bytes)\n", filePath, input_size);

    // 复制输入数据到设备
    unsigned char *d_input;
    cudaMalloc((void**)&d_input, input_size * sizeof(unsigned char));
    cudaMemcpy(d_input, testStr, input_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int num_chunks = (input_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // 分配临时压缩输出缓冲区，每块预留 MAX_COMPRESSED_CHUNK_SIZE 字节
    unsigned char *d_tempOut;
    cudaMalloc((void**)&d_tempOut, num_chunks * MAX_COMPRESSED_CHUNK_SIZE * sizeof(unsigned char));

    // 分配每块压缩后总bit数数组
    int *d_chunkBitSizes;
    cudaMalloc((void**)&d_chunkBitSizes, num_chunks * sizeof(int));

    int sharedMemSize = CHUNK_SIZE * sizeof(int);
    compressKernelBitPack<<<num_chunks, 256, sharedMemSize>>>(d_input, input_size, CHUNK_SIZE, d_tempOut, d_chunkBitSizes);
    cudaDeviceSynchronize();

    // 分配并计算各压缩块在最终输出中的字节偏移
    int *d_chunkByteOffsets;
    cudaMalloc((void**)&d_chunkByteOffsets, (num_chunks + 1) * sizeof(int));
    computeOffsetsKernel<<<1, 1>>>(d_chunkBitSizes, d_chunkByteOffsets, num_chunks);
    cudaDeviceSynchronize();

    int h_finalSize;
    cudaMemcpy(&h_finalSize, d_chunkByteOffsets + num_chunks, sizeof(int), cudaMemcpyDeviceToHost);
    printf("压缩后总字节数: %d bytes\n", h_finalSize);

    unsigned char *d_finalOut;
    cudaMalloc((void**)&d_finalOut, h_finalSize * sizeof(unsigned char));
    assembleKernelBitPack<<<num_chunks, 256>>>(d_tempOut, d_chunkByteOffsets, num_chunks, d_finalOut);
    cudaDeviceSynchronize();

    // 将最终压缩数据拷回主机并写入文件
    unsigned char *h_finalOut = (unsigned char *)malloc(h_finalSize * sizeof(unsigned char));
    cudaMemcpy(h_finalOut, d_finalOut, h_finalSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();

    // 计算经过的时间（单位：毫秒）
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 使用 printf 输出，保留两位小数
    printf("压缩耗时: %ld ms\n", duration.count());

    FILE *file = fopen("compressed_output.bin", "wb");
    if (file) {
        fwrite(h_finalOut, sizeof(unsigned char), h_finalSize, file);
        fclose(file);
        printf("压缩数据已成功写入文件 compressed_output.bin\n");
    } else {
        printf("无法打开文件进行写入\n");
    }

    // 记录开始时间
    start = std::chrono::high_resolution_clock::now();
    // 分配解压输出缓冲区
    unsigned char *d_decompOut;
    cudaMalloc((void**)&d_decompOut, input_size * sizeof(unsigned char));

    int *d_decompSize;
    cudaMalloc((void**)&d_decompSize, sizeof(int));
    cudaMemset(d_decompSize, 0, sizeof(int));
    // 注意：调用 decompressKernelBitPack 时参数顺序必须与其定义完全一致
    decompressKernelBitPack<<<num_chunks, 256, sizeof(int)>>>(d_finalOut, d_chunkBitSizes, d_chunkByteOffsets, CHUNK_SIZE, d_decompOut, input_size, d_decompSize);
    cudaDeviceSynchronize();

    unsigned char *h_decompOut = (unsigned char*)malloc(input_size * sizeof(unsigned char));
    cudaMemcpy(h_decompOut, d_decompOut, input_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // printf("解压后数据:\n%s\n", h_decompOut);
    
    int h_decompSize;
    cudaMemcpy(&h_decompSize, d_decompSize, sizeof(int), cudaMemcpyDeviceToHost);
    printf("解压后的数据字节数: %d bytes\n", h_decompSize);
    cudaFree(d_decompSize);

    // 记录结束时间
    end = std::chrono::high_resolution_clock::now();

    // 计算经过的时间（单位：毫秒）
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 使用 printf 输出，保留两位小数
    printf("解压缩耗时: %ld ms\n", duration.count());

    // 清理内存
    free(h_finalOut);
    free(h_decompOut);
    free(testStr);
    cudaFree(d_input);
    cudaFree(d_tempOut);
    cudaFree(d_chunkBitSizes);
    cudaFree(d_chunkByteOffsets);
    cudaFree(d_finalOut);
    cudaFree(d_decompOut);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("请提供一个目录路径作为参数。\n");
        return 1;
    }
    const char *directoryPath = argv[1];
    DIR *dir = opendir(directoryPath);
    if (!dir) {
        printf("无法打开目录: %s\n", directoryPath);
        return 1;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        char filePath[512];
        snprintf(filePath, sizeof(filePath), "%s/%s", directoryPath, entry->d_name);
        processFile(filePath);
    }
    closedir(dir);
    return 0;
}