#pragma once

#include "matrix.h"

Matrix** load_csv(char* csv_file, bool header);
Matrix* load_ubyte_images(char* ubyte_file);
Matrix* load_ubyte_labels(char* ubyte_file);
int count_samples(char* csv_file, bool header);
int count_features(char *filename, bool header);
