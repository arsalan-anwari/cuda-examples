#include <array>
#include <random>
#include <iterator>
#include <algorithm>
#include <utility>
#include <iostream>
#include <string_view>

template<class T>
__global__ void vec_sum(T* a, T* b, T* c, T size){
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (idx < size){
        c[idx] = a[idx] + b[idx];
    }
};

template<class Container>
void fill_rand_range( Container& data, typename Container::value_type min, typename Container::value_type max ){
    auto start = data.begin();
    auto end = data.end();

    static std::random_device rnd;    // you only need to initialize it once
    static std::mt19937 mte(rnd());   // this is a relative big object to create

    std::uniform_int_distribution<typename Container::value_type> dist(min, max);
    std::generate(start, end, [&] () { return dist(mte); });
};

template<class Container>
void print_container( const Container& data, const std::string_view name ){
    std::cout << name << " [" << data.size() << "] = {";
    for(const int elem: data) std::cout << elem << ", ";
    std::cout
    << "[min = " << *std::min_element(data.begin(), data.end())
    << ", max = " << *std::max_element(data.begin(), data.end())
    << ", mean = " << (std::accumulate(data.begin(), data.end(), 0) / data.size())
    << "]}" << std::endl;
};

int main(){

    using ELEM_TYPE = int;
    constexpr size_t ELEM_SIZE = 128;
    constexpr size_t ELEM_BYTE_SIZE = ELEM_SIZE * sizeof(ELEM_TYPE);

    std::array<ELEM_TYPE, ELEM_SIZE> h_a, h_b, h_c;
    ELEM_TYPE *d_a, *d_b, *d_c;

    // Fill data with random values on CPU
    fill_rand_range( h_a, 0, 1000 );
    fill_rand_range( h_b, 0, 1000 );

    // Show input data
    print_container(h_a, "h_a");
    print_container(h_b, "h_b");

    // Allocate memory on GPU
    cudaMalloc( reinterpret_cast<void**>(&d_a), ELEM_BYTE_SIZE );
    cudaMalloc( reinterpret_cast<void**>(&d_b), ELEM_BYTE_SIZE );
    cudaMalloc( reinterpret_cast<void**>(&d_c), ELEM_BYTE_SIZE );

    // Copy CPU data to GPU
    cudaMemcpy( d_a, h_a.data(), ELEM_BYTE_SIZE, cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, h_b.data(), ELEM_BYTE_SIZE, cudaMemcpyHostToDevice );

    // Launch GPU kernel
    vec_sum<int><<<(ELEM_SIZE/32), 32>>>(d_a, d_b, d_c, ELEM_SIZE);
    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(h_c.data(), d_c, ELEM_BYTE_SIZE, cudaMemcpyDeviceToHost);

    // Show results
    print_container(h_c, "h_c");

    // Free device memmory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}