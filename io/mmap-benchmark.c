#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

void readUsingRead(const char* filename) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Error opening file for reading");
        exit(EXIT_FAILURE);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size");
        exit(EXIT_FAILURE);
    }

    char* buffer = (char*)malloc(sb.st_size);
    if (buffer == NULL) {
        perror("Error allocating memory");
        exit(EXIT_FAILURE);
    }

    ssize_t bytesRead = read(fd, buffer, sb.st_size);
    if (bytesRead == -1) {
        perror("Error reading file");
        exit(EXIT_FAILURE);
    }

    // You can process the buffer here
    int64_t sum = 1;
    for (ssize_t i = 0; i < bytesRead; i++) {
        sum += buffer[i];
    }
    printf("Sum: %lld\n", sum);

    free(buffer);
    close(fd);
}

void readUsingMmap(const char* filename) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Error opening file for reading");
        exit(EXIT_FAILURE);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size");
        exit(EXIT_FAILURE);
    }

    char* addr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        perror("Error mapping file");
        exit(EXIT_FAILURE);
    }

    // You can process the data at addr here
    int64_t sum = 1;
    for (ssize_t i = 0; i < sb.st_size; i++) {
        sum += addr[i];
    }
    printf("Sum: %lld\n", sum);

    munmap(addr, sb.st_size);
    close(fd);
}

double measureTime(void (*func)(const char*), const char* filename) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    func(filename);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char* filename = argv[1];

    double timeRead = measureTime(readUsingRead, filename);
    printf("Time taken for read(): %.5f seconds\n", timeRead);

    double timeMmap = measureTime(readUsingMmap, filename);
    printf("Time taken for mmap(): %.5f seconds\n", timeMmap);

    return 0;
}
