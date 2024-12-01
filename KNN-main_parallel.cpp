#include <mpi.h>
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

// Main MPI function
int main(int argc, char **argv) {
    srand(static_cast<unsigned>(time(0)));
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Record start time
    double start_time = MPI_Wtime();

    int training_size = NUM_CLUSTERS * POINTS_PER_CLUSTER;
    int chunk_size = (training_size + size - 1) / size; // Ensure all processes get data

    vector<float> training_data; // Flattened training data
    vector<int> training_labels; // Training labels
    vector<float> test_data(TEST_POINTS * NUM_FEATURES, 0.0); // Flattened test data

    if (rank == 0) {
        // Generate and shuffle training data
        generate_clusters(training_data, training_labels);
        shuffle_data(training_data, training_labels, NUM_FEATURES);

        normalize_data(training_data, NUM_FEATURES);

        for (int i = 0; i < TEST_POINTS * NUM_FEATURES; i++) {
            test_data[i] = static_cast<float>((rand() % 100) - 50);
        }
        normalize_data(test_data, NUM_FEATURES);

        // Pad training data and labels
        training_data.resize(chunk_size * size * NUM_FEATURES, 0.0);
        training_labels.resize(chunk_size * size, -1);

        // Adding MPI_Send to send the data to all processes (Rank 0 will also act as a sender)
        for (int i = 1; i < size; i++) {
            MPI_Send(training_data.data(), chunk_size * NUM_FEATURES, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            MPI_Send(training_labels.data(), chunk_size, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(test_data.data(), TEST_POINTS * NUM_FEATURES, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
        }
    } else {
        // Allocate memory for training data and labels before receiving
        training_data.resize(chunk_size * NUM_FEATURES, 0.0);
        training_labels.resize(chunk_size, -1);
        test_data.resize(TEST_POINTS * NUM_FEATURES, 0.0);

        // Debugging print statements
        cout << "Process " << rank << " is about to receive training data and test data.\n";

        // Receive the training data and labels for each process
        MPI_Recv(training_data.data(), chunk_size * NUM_FEATURES, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(training_labels.data(), chunk_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(test_data.data(), TEST_POINTS * NUM_FEATURES, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Barrier to ensure synchronization between processes before starting the prediction
    MPI_Barrier(MPI_COMM_WORLD);

    // Predict labels for test points
    int k = 5; // Number of neighbors
    vector<int> local_predictions(TEST_POINTS);
    vector<float> local_distances(TEST_POINTS);

    for (int i = 0; i < TEST_POINTS; i++) {
        local_predictions[i] = knn_predict(training_data.data(), training_labels.data(), chunk_size,
                                           &test_data[i * NUM_FEATURES], NUM_FEATURES, k);
        local_distances[i] = calculate_distance(&training_data[0], &test_data[i * NUM_FEATURES], NUM_FEATURES);
    }

    // Additional MPI calls within this loop
    for (int i = 0; i < TEST_POINTS; i++) {
        MPI_Send(&local_predictions[i], 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
        MPI_Send(&local_distances[i], 1, MPI_FLOAT, 0, 11, MPI_COMM_WORLD);
    }

    // Gather results at rank 0
    vector<int> global_predictions(TEST_POINTS * size);
    vector<float> global_distances(TEST_POINTS * size);

    MPI_Gather(local_predictions.data(), TEST_POINTS, MPI_INT, global_predictions.data(),
               TEST_POINTS, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_distances.data(), TEST_POINTS, MPI_FLOAT, global_distances.data(),
               TEST_POINTS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Additional MPI calls for gathering accuracy and distances (subdivide work further)
    if (rank == 0) {
        for (int i = 0; i < TEST_POINTS; i++) {
            cout << "Test Point " << i << ": Predicted Cluster = "
                 << global_predictions[i] << ", Min Distance = "
                 << global_distances[i] << endl;
        }

        // Record end time
        double end_time = MPI_Wtime();
        double execution_time = end_time - start_time;

        // Display execution time
        cout << "Execution Time: " << execution_time << " seconds" << endl;
    }

    // Free allocated memory
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes complete before finishing
    MPI_Finalize();
    return 0;
}
