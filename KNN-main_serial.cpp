#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm> // For std::shuffle
#include <random>    // For std::default_random_engine

using namespace std;

#define NUM_CLUSTERS 16        // Number of clusters increased to 8
#define POINTS_PER_CLUSTER 2000 // Points per cluster
#define TEST_POINTS 700000        // Number of test points
#define NUM_FEATURES 4       // Increased to 4 features for more complexity

// Function to generate balanced clusters
void generate_clusters(vector<float> &data, vector<int> &labels) {
    for (int cluster = 0; cluster < NUM_CLUSTERS; cluster++) {
        float center_x = static_cast<float>((cluster + 1) * 10); // Spread clusters in x-axis
        float center_y = static_cast<float>((cluster + 1) * 10); // Spread clusters in y-axis
        for (int i = 0; i < POINTS_PER_CLUSTER; i++) {
            data.push_back(static_cast<float>(center_x + (rand() % 10) - 5));
            data.push_back(static_cast<float>(center_y + (rand() % 10) - 5));
            data.push_back(static_cast<float>((rand() % 10) - 5)); // Added 3rd feature
            data.push_back(static_cast<float>((rand() % 10) - 5)); // Added 4th feature
            labels.push_back(cluster);
        }
    }
}

// Function to shuffle training data and labels together
void shuffle_data(vector<float> &data, vector<int> &labels, int num_features) {
    int num_points = data.size() / num_features;
    vector<int> indices(num_points);
    for (int i = 0; i < num_points; i++) {
        indices[i] = i;
    }

    // Create a random number generator
    random_device rd;
    default_random_engine rng(rd());

    // Shuffle indices
    shuffle(indices.begin(), indices.end(), rng);

    // Create shuffled data and labels
    vector<float> shuffled_data(data.size());
    vector<int> shuffled_labels(labels.size());
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_features; j++) {
            shuffled_data[i * num_features + j] = data[indices[i] * num_features + j];
        }
        shuffled_labels[i] = labels[indices[i]];
    }

    // Replace original data and labels with shuffled versions
    data = shuffled_data;
    labels = shuffled_labels;
}

// Function to normalize data
void normalize_data(vector<float> &data, int num_features) {
    int num_points = data.size() / num_features;
    for (int j = 0; j < num_features; j++) {
        float mean = 0.0, std_dev = 0.0;
        for (int i = 0; i < num_points; i++) {
            mean += data[i * num_features + j];
        }
        mean /= num_points;

        for (int i = 0; i < num_points; i++) {
            std_dev += (data[i * num_features + j] - mean) * (data[i * num_features + j] - mean);
        }
        std_dev = sqrt(std_dev / num_points);

        for (int i = 0; i < num_points; i++) {
            data[i * num_features + j] = (data[i * num_features + j] - mean) / (std_dev + 1e-8);
        }
    }
}

// Function to calculate Euclidean distance
float calculate_distance(const float *point1, const float *point2, int num_features) {
    float distance = 0.0;
    for (int i = 0; i < num_features; i++) {
        distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sqrt(distance);
}

// Function to predict the label of a test point
int knn_predict(const float *training_data, const int *training_labels, int training_size,
                const float *test_point, int num_features, int k) {
    vector<pair<float, int>> distances;
    for (int i = 0; i < training_size; i++) {
        float dist = calculate_distance(&training_data[i * num_features], test_point, num_features);
        distances.push_back({dist, training_labels[i]});
    }
    sort(distances.begin(), distances.end());

    // Count neighbors
    vector<int> neighbor_counts(NUM_CLUSTERS, 0);
    for (int i = 0; i < k; i++) {
        neighbor_counts[distances[i].second]++;
    }

    return max_element(neighbor_counts.begin(), neighbor_counts.end()) - neighbor_counts.begin();
}

// Main function
int main() {
    srand(static_cast<unsigned>(time(0)));

    // Record start time
    double start_time = static_cast<double>(clock()) / CLOCKS_PER_SEC;

    int training_size = NUM_CLUSTERS * POINTS_PER_CLUSTER;
    vector<float> training_data; // Flattened training data
    vector<int> training_labels; // Training labels
    vector<float> test_data(TEST_POINTS * NUM_FEATURES, 0.0); // Flattened test data

    // Generate and shuffle training data
    generate_clusters(training_data, training_labels);
    shuffle_data(training_data, training_labels, NUM_FEATURES);
    normalize_data(training_data, NUM_FEATURES);

    for (int i = 0; i < TEST_POINTS * NUM_FEATURES; i++) {
        test_data[i] = static_cast<float>((rand() % 100) - 50);
    }
    normalize_data(test_data, NUM_FEATURES);

    // Predict labels for test points
    int k = 5; // Number of neighbors
    vector<int> predictions(TEST_POINTS);
    vector<float> distances(TEST_POINTS);

    for (int i = 0; i < TEST_POINTS; i++) {
        predictions[i] = knn_predict(training_data.data(), training_labels.data(), training_size,
                                      &test_data[i * NUM_FEATURES], NUM_FEATURES, k);
        distances[i] = calculate_distance(&training_data[0], &test_data[i * NUM_FEATURES], NUM_FEATURES);
    }

    // Output results
    for (int i = 0; i < TEST_POINTS; i++) {
        cout << "Test Point " << i << ": Predicted Cluster = "
             << predictions[i] << ", Min Distance = "
             << distances[i] << endl;
    }

    // Record end time
    double end_time = static_cast<double>(clock()) / CLOCKS_PER_SEC;
    double execution_time = end_time - start_time;

    // Display execution time
    cout << "Execution Time: " << execution_time << " seconds" << endl;

    return 0;
}
