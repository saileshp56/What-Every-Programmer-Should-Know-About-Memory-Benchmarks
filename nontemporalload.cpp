#include <iostream>
#include <chrono>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>
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

    vector<float> data = genRandData(dataSize);

    float* aligned_data = (float*)aligned_alloc(16, dataSize * sizeof(float));
    std::copy(data.begin(), data.end(), aligned_data);

    __m128i* aligned_int_data = (__m128i*)aligned_alloc(16, dataSize / 4 * sizeof(__m128i));
    for (size_t i = 0; i < dataSize / 4; ++i) {
        aligned_int_data[i] = _mm_set_epi32(i, i+1, i+2, i+3);
    }
    
    double countStore = benchmark("_mm_load_ps", [&]() {
        __m128 temp;
        for (size_t i = 0; i < dataSize; i += 4) {
            temp = _mm_load_ps(&aligned_data[i]);
        }
    });

    double countStream = benchmark("_mm_stream_load_si128", [&]() {
        __m128i temp;
        for (size_t i = 0; i < dataSize / 4; i++)
        {
            temp = _mm_stream_load_si128(&aligned_int_data[i]);
        }
    });

    if (countStream < countStore) {
        cout << "Non-temporal (Streaming Load Buffer) is faster by " << (countStore - countStream) << " seconds" << endl;
    } else {
        cout << "Temporal (Cache) is faster by " << (countStream - countStore) << " seconds" << endl;
    }

    free(aligned_data);
    free(aligned_int_data);

    return 0;
}

/*
    Cache is faster than buffer, so you should really only use this 
    if you’re trying to avoid cache pollution,
    otherwise you will be slower using stream_load

  If you have compiler errors try this: g++ -O2 -msse4.1 nontemporalload.cpp -o nontemporalload
*/
