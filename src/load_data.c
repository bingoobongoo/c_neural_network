#include "load_data.h"

#define MAX_LINE_LEN 32768

Matrix** load_csv(char* csv_file, bool header) {
    int n_features = count_features(csv_file, header);
    int n_samples = count_samples(csv_file, header);
    FILE* file = fopen(csv_file, "r");
    if (!file) {
        perror("File not found");
        exit(1);
    }

    Matrix** loaded_data = (Matrix**)malloc(2 * sizeof(Matrix*));
    Matrix* feature_m = matrix_new(n_samples, n_features);
    Matrix* label_m = matrix_new(n_samples, 1);

    char line[MAX_LINE_LEN];

    if (header) {
        fgets(line, sizeof(line), file);
    }

    int i = 0;
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = 0;
        char* token = strtok(line, ",");
        char* endptr;
        int j = 0;
        while (token != NULL) {
            if (j == 0) {
                matrix_assign(label_m, i, 0, strtod(token, &endptr));
            }
            else {
                matrix_assign(feature_m, i, j-1, strtod(token, &endptr));
            }
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }

    fclose(file);

    loaded_data[0] = feature_m;
    loaded_data[1] = label_m;

    return loaded_data;
}

Matrix* load_ubyte_images(char* ubyte_file) {
    FILE* file = fopen(ubyte_file, "rb");
    if (!file) {
        perror("File not found");
        exit(1);
    }

    unsigned char header[16];
    fread(header, sizeof(header), 1, file);
    int magic = (header[0] << 24) | (header[1] << 16) | (header[2] << 8) | header[3];
    int n_samples = (header[4] << 24) | (header[5] << 16) | (header[6] << 8) | header[7];
    int height = (header[8] << 24) | (header[9] << 16) | (header[10] << 8) | header[11];
    int width = (header[12] << 24) | (header[13] << 16) | (header[14] << 8) | header[15];

    if (magic != 0x00000803) {
        fprintf(stderr, "Invalid magic number: %#x (expected 0x00000803)\n", magic);
        fclose(file);
        exit(1);
    }

    int n_pixels = height * width;

    Matrix* feature_m = matrix_new(n_samples, n_pixels);
    unsigned char* pixel_values = malloc(n_pixels * sizeof(unsigned char));
    for (int i=0; i<n_samples; i++) {
        fread(pixel_values, 1, n_pixels, file);
        for (int j=0; j<n_pixels; j++) {
            matrix_assign(feature_m, i, j, (double)pixel_values[j]);
        }
    }

    free(pixel_values);
    fclose(file);

    return feature_m;
}

Matrix* load_ubyte_labels(char* ubyte_file) {
    FILE* file = fopen(ubyte_file, "rb");
    if (!file) {
        perror("File not found");
        exit(1);
    }

    unsigned char header[8];
    fread(header, sizeof(header), 1, file);
    int magic = (header[0] << 24) | (header[1] << 16) | (header[2] << 8) | header[3];
    int n_labels = (header[4] << 24) | (header[5] << 16) | (header[6] << 8) | header[7];

    if (magic != 0x00000801) {
        fprintf(stderr, "Invalid magic number: %#x (expected 0x00000803)\n", magic);
        fclose(file);
        exit(1);
    }

    Matrix* label_m = matrix_new(n_labels, 1);
    unsigned char* labels = malloc(n_labels * sizeof(unsigned char));
    fread(labels, 1, n_labels, file);
    for (int i=0; i<n_labels; i++) {
        matrix_assign(label_m, i, 0, (double)labels[i]);
    }

    free(labels);
    fclose(file);

    return label_m;
}

int count_samples(char* csv_file, bool header) {
    FILE* file = fopen(csv_file, "r");
    if (!file) {
        perror("File not found");
        exit(1);
    }

    int lines = 0;
    int ch;

    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') lines++;
    }

    if (header) lines--;

    fclose(file);

    return lines;

}

int count_features(char *filename, bool header) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("File opening failed");
        exit(1);
    }

    char line[MAX_LINE_LEN];

    if (header) {
        fgets(line, sizeof(line), file);
    }

    int count = 1;
    for (int i = 0; line[i] != '\0'; i++) {
        if (line[i] == ',') count++;
        if (line[i] == '\n') break;
    }

    fclose(file);

    return count;
}