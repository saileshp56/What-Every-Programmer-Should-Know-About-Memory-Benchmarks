#include <iostream>
#include <chrono>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <vector>
#include <random>
#include <cstdlib> 

using namespace std;
using namespace std::chrono;

// Generate random float data
vector<float> genRandData(size_t size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> distrib(0.0f, 1.0f);
    vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = distrib(gen);
    }
    return data;
}

auto benchmark(const string& name, auto func) -> double {
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    duration<double> durationSec = end - start;

    auto nano = duration_cast<nanoseconds>(end - start);

    cout << name << ": " << nano.count() << " nanoseconds" << endl;

    cout << name << ": " << durationSec.count() << " seconds" << endl;
    return durationSec.count();
}

int main() {
    const size_t dataSize = 32768 * 32768;
    // const size_t dataSize = 2;

    vector<float> data = genRandData(dataSize);
    // Allocate aligned memory
    float* aligned_data = (float*)aligned_alloc(16, dataSize * sizeof(float));
    float* aligned_result_store = (float*)aligned_alloc(16, dataSize * sizeof(float));
    float* aligned_result_stream = (float*)aligned_alloc(16, dataSize * sizeof(float));

    copy(data.begin(), data.end(), aligned_data);

    // Benchmark _mm_store_ps
    double countStore = benchmark("_mm_store_ps", [&]() {
        for (size_t i = 0; i < dataSize; i += 4) {
            __m128 a = _mm_load_ps(&aligned_data[i]);
            _mm_store_ps(&aligned_result_store[i], a);
        }
    });

    // Benchmark _mm_stream_ps
    double countStream = benchmark("_mm_stream_ps", [&]() {
        for (size_t i = 0; i < dataSize; i += 4) {
            __m128 a = _mm_load_ps(&aligned_data[i]);
            _mm_stream_ps(&aligned_result_stream[i], a);
        }
    });
    // (1) stream avoids reading cache (load may read it)
    // (2) avoids writing to cache, and uses write-combining

    // You can try this too to fill the size of a whole cache line after reading
    // though depending on compilers it might end up being the same due to re-ordering
    // reordering is relaxed with NT
    
    // double countStream = benchmark("_mm_stream_ps", [&]() {
    //     for (size_t i = 0; i < dataSize; i += 16) {
    //         __m128 a = _mm_load_ps(&aligned_data[i]);
    //         __m128 b = _mm_load_ps(&aligned_data[i + 4]);
    //         __m128 c = _mm_load_ps(&aligned_data[i + 8]);
    //         __m128 d = _mm_load_ps(&aligned_data[i + 12]);


    //         _mm_stream_ps(&aligned_result_stream[i], a);
    //         _mm_stream_ps(&aligned_result_stream[i + 4], b);
    //         _mm_stream_ps(&aligned_result_stream[i + 8], c);
    //         _mm_stream_ps(&aligned_result_stream[i + 12], d);

    //     }
    // });
    
    if (countStream < countStore) {
        cout << "Stream is faster by " << (countStore - countStream) << " seconds" << endl;
    } else {
        cout << "Store is faster by " << (countStream - countStore) << " seconds" << endl;
    }

    free(aligned_data);
    free(aligned_result_store);
    free(aligned_result_stream);

    return 0;
}
