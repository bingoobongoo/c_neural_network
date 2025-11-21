#pragma once

#include "config.h"
#include "matrix.h"

#define CIFAR_IMAGE_BYTES 3072
#define CIFAR_RECORD_BYTES 3073

Matrix** load_csv(char* csv_file, bool header);
Matrix* load_ubyte_images(char* ubyte_file);
Matrix* load_ubyte_labels(char* ubyte_file);
Matrix* load_cifar10_images(const char** bin_files, int n_files);
Matrix* load_cifar10_labels(const char** bin_files, int n_files);
int count_samples(char* csv_file, bool header);
int count_features(char *filename, bool header);
