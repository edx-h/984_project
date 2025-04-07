#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <chrono>

// 定义常量
#define CHUNK_SIZE 1024
#define MAX_COMPRESSED_CHUNK_SIZE (CHUNK_SIZE)
#define DICT_SIZE 16

// 字典：与GPU版本一致
const unsigned char dictionary[DICT_SIZE] = { ' ', 'e', 't', 'a', 'o', 'i', 'n', 's', 'r', 'h', 'l', 'd', 'u', 'c', 'm', 'f' };

// 读取文件内容
std::vector<unsigned char> readFile(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return {};
    }
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();
    return buffer;
}

// 写入文件
void writeFile(const std::string& filePath, const std::vector<unsigned char>& data) {
    std::ofstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件进行写入: " << filePath << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    file.close();
}

// 压缩函数
std::vector<unsigned char> compress(const std::vector<unsigned char>& input) {
    std::vector<unsigned char> compressed;
    int bitPos = 0;              // 当前字节中的位偏移
    unsigned char currentByte = 0; // 当前正在填充的字节

    for (unsigned char symbol : input) {
        bool found = false;
        int dict_index = 0;
        // 检查符号是否在字典中
        for (int d = 0; d < DICT_SIZE; d++) {
            if (symbol == dictionary[d]) {
                found = true;
                dict_index = d;
                break;
            }
        }
        if (found) {
            // 编码为5位：1位标记（1）+ 4位索引
            int code = (1 << 4) | (dict_index & 0xF);
            int bits = 5;
            for (int i = 0; i < bits; i++) {
                if (bitPos == 8) {
                    compressed.push_back(currentByte);
                    currentByte = 0;
                    bitPos = 0;
                }
                currentByte |= ((code >> (bits - 1 - i)) & 1) << (7 - bitPos);
                bitPos++;
            }
        } else {
            // 编码为9位：1位标记（0）+ 8位原始符号
            int code = symbol;
            int bits = 9;
            for (int i = 0; i < bits; i++) {
                if (bitPos == 8) {
                    compressed.push_back(currentByte);
                    currentByte = 0;
                    bitPos = 0;
                }
                currentByte |= ((code >> (bits - 1 - i)) & 1) << (7 - bitPos);
                bitPos++;
            }
        }
    }
    // 处理最后一个未满的字节
    if (bitPos > 0) {
        compressed.push_back(currentByte);
    }
    return compressed;
}

// 解压缩函数
std::vector<unsigned char> decompress(const std::vector<unsigned char>& compressed, size_t originalSize) {
    std::vector<unsigned char> decompressed;
    size_t bitPos = 0;    // 全局位偏移
    size_t byteIndex = 0; // 当前字节索引

    while (decompressed.size() < originalSize && byteIndex < compressed.size()) {
        // 读取1位标记
        unsigned char currentByte = compressed[byteIndex];
        int flag = (currentByte >> (7 - (bitPos % 8))) & 1;
        bitPos++;
        if (bitPos == 8) {
            byteIndex++;
            bitPos = 0;
        }

        if (flag == 1) {
            // 读取4位索引
            int index = 0;
            for (int i = 0; i < 4; i++) {
                if (byteIndex >= compressed.size()) break;
                currentByte = compressed[byteIndex];
                index = (index << 1) | ((currentByte >> (7 - (bitPos % 8))) & 1);
                bitPos++;
                if (bitPos == 8) {
                    byteIndex++;
                    bitPos = 0;
                }
            }
            if (index < DICT_SIZE) {
                decompressed.push_back(dictionary[index]);
            }
        } else {
            // 读取8位原始符号
            unsigned char symbol = 0;
            for (int i = 0; i < 8; i++) {
                if (byteIndex >= compressed.size()) break;
                currentByte = compressed[byteIndex];
                symbol = (symbol << 1) | ((currentByte >> (7 - (bitPos % 8))) & 1);
                bitPos++;
                if (bitPos == 8) {
                    byteIndex++;
                    bitPos = 0;
                }
            }
            decompressed.push_back(symbol);
        }
    }
    return decompressed;
}

// 处理文件
void processFile(const std::string& filePath) {
    auto start = std::chrono::high_resolution_clock::now();

    // 读取文件
    std::vector<unsigned char> input = readFile(filePath);
    if (input.empty()) return;

    std::cout << "正在处理文件: " << filePath << " (大小: " << input.size() << " bytes)" << std::endl;

    // 压缩
    std::vector<unsigned char> compressed = compress(input);
    std::cout << "压缩后总字节数: " << compressed.size() << " bytes" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "压缩耗时: " << duration.count() << " ms" << std::endl;

    // 写入压缩文件
    writeFile("compressed_output.bin", compressed);
    std::cout << "压缩数据已成功写入文件 compressed_output.bin" << std::endl;

    // 解压缩
    start = std::chrono::high_resolution_clock::now();
    std::vector<unsigned char> decompressed = decompress(compressed, input.size());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "解压后的数据字节数: " << decompressed.size() << " bytes" << std::endl;
    std::cout << "解压缩耗时: " << duration.count() << " ms" << std::endl;

    // 验证
    if (decompressed == input) {
        std::cout << "解压数据与原始数据一致" << std::endl;
    } else {
        std::cout << "解压数据与原始数据不一致" << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "请提供一个文件路径作为参数。" << std::endl;
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