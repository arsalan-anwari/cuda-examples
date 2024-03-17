#include <array>
#include <random>
#include <iterator>
#include <algorithm>
#include <utility>
#include <iostream>
#include <string_view>
#include <concepts>

// template<class T>
// __global__ void vec_sum(const T* a, const T* b, T* c, T size){
//     int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
//     if (idx < size){
//         c[idx] = a[idx] + b[idx];
//     }
// };

template<std::integral T>
struct range_t{ T start; T end; };


template<class Container>
void fill_rand_range( Container& data, const range_t<typename Container::value_type>& range ){
    auto start = data.begin();
    auto end = data.end();

    static std::random_device rnd;    // you only need to initialize it once
    static std::mt19937 mte(rnd());   // this is a relative big object to create

    std::uniform_int_distribution<typename Container::value_type> dist(range.start, range.end);
    std::generate(start, end, [&] () { return dist(mte); });
    
}

template<class Container>
void print_matrix( const Container& data, const std::string_view name, size_t stride ){
    size_t col = 0, row = 0;

    std::cout << name << " " << "[" << stride << "]" << "[" << stride << "]" << " = {" << std::endl;
    
    std::cout << "[" << row++ << "] ";
    for(const int elem: data){ 
        std::cout << elem << " ";
        col++;

        if (col > (stride - 1)) { 
            std::cout << std::endl;
            if (row < stride) { std::cout << "[" << row++ << "] "; }
            col = 0; 
        }
    }
    std::cout << "}" << std::endl;
}



template<typename T, unsigned CHUNK_SIZE, unsigned WIDTH> requires (std::integral<T> || std::floating_point<T>)
__global__ void matrix_mul(const T* a, const T* b, T* c){

    __shared__ T chunk_a[CHUNK_SIZE * CHUNK_SIZE]; // used to store chunk of rows in A
    __shared__ T chunk_b[CHUNK_SIZE * CHUNK_SIZE]; // used to store chunk of columns in B

    unsigned row = (blockDim.y * blockIdx.y) + threadIdx.y;
    unsigned col = (blockDim.x * blockIdx.x) + threadIdx.x;
    T chunk_sum = 0;

    // For the current thread in the chunk loop through all rows of A and columns from B
    // You will not go out of bounds as you will always start from the top most  
    for(unsigned block_offset = 0; block_offset < WIDTH; block_offset += CHUNK_SIZE){
        
        // For each thread in the chunk calculate its absolute offset in memory and store in a temporary buffer
        // Example: 
        //  - A[1][1] = A[5]  -> {CHUNK_SIZE := 2, block_offset := 0}
        //  - A[1][1] = A[7]  -> {CHUNK_SIZE := 2, block_offset := 2}
        //  - B[1][1] = A[5]  -> {CHUNK_SIZE := 2, block_offset := 0}
        //  - B[1][1] = A[13] -> {CHUNK_SIZE := 2, block_offset := 2}
        unsigned chunk_a_idx = (row * WIDTH) + (block_offset * CHUNK_SIZE) + threadIdx.x;
        unsigned chunk_b_idx = (block_offset * WIDTH) + (threadIdx.y * WIDTH) + col; 

        // For each thread in the chunk, store the data in the shared memory.
        // Mapping: chunk_a[ty * CHUNK_SIZE + tx] == chunk_a[ty][tx]
        chunk_a[(threadIdx.y * CHUNK_SIZE) + threadIdx.x] = a[chunk_a_idx]; 
        chunk_b[(threadIdx.y * CHUNK_SIZE) + threadIdx.x] = b[chunk_b_idx];
        __syncthreads(); // wait untill all threads in chunk have stored one element before calculating dot product

        // For each thread calculate the dot product from the chunk stored in shared memory based on the threadIdx.
        // For each iteration of block_offset, different values will be loaded and summed, where the total sum of all chunks
        // is the value stored at the output matrix.
        // Example {CHUNK_SIZE := 2}: 
        //  - chunk_sum += (chunk_a[1][0] * chunk_b[0][1]) + (chunk_a[1][1] * chunk_b[1][1])
        for(unsigned i = 0; i < blockDim.x; i++){
            chunk_sum += chunk_a[(threadIdx.y * CHUNK_SIZE) + i] * chunk_b[(i * CHUNK_SIZE) + threadIdx.x];
        }
        __syncthreads();
    }

    // Store result of chunk in element c[row][col];
    c[(row * WIDTH) + col] = chunk_sum;
}


int main(){

    // Define project settings {change based on your hardware and compiler settings}
    // Current settings: Matrix {512 x 512}, Chunk {32 x 32} -> ~ 1MB stackframe
    using ELEM_TYPE = int;
    constexpr size_t MATRIX_WIDTH = 1 << 9;
    constexpr size_t CHUNK_SIZE = 32;
    constexpr size_t MATRIX_SIZE = MATRIX_WIDTH * MATRIX_WIDTH;
    constexpr size_t MATRIX_SIZE_BYTES = MATRIX_SIZE * sizeof(ELEM_TYPE);

    // Create storage containers
    std::array<ELEM_TYPE, MATRIX_SIZE> h_a, h_b, h_c;
    ELEM_TYPE *d_a, *d_b, *d_c;

    // Fill data with random values on CPU
    fill_rand_range( h_a, {1, 10} );
    fill_rand_range( h_b, {1, 10} );

    // Show input data
    // print_matrix(h_a, "h_a", MATRIX_WIDTH);
    // print_matrix(h_b, "h_b", MATRIX_WIDTH);

    // Allocate memory on GPU
    cudaMalloc( reinterpret_cast<void**>(&d_a), MATRIX_SIZE_BYTES );
    cudaMalloc( reinterpret_cast<void**>(&d_b), MATRIX_SIZE_BYTES );
    cudaMalloc( reinterpret_cast<void**>(&d_c), MATRIX_SIZE_BYTES );

    // Copy CPU data to GPU
    cudaMemcpy( d_a, h_a.data(), MATRIX_SIZE_BYTES, cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, h_b.data(), MATRIX_SIZE_BYTES, cudaMemcpyHostToDevice );

    // Launch GPU kernel
    dim3 grid(MATRIX_WIDTH / CHUNK_SIZE, MATRIX_WIDTH / CHUNK_SIZE);
    dim3 threads(CHUNK_SIZE, CHUNK_SIZE);

    matrix_mul<int, CHUNK_SIZE, MATRIX_WIDTH><<<grid, threads>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(h_c.data(), d_c, MATRIX_SIZE_BYTES, cudaMemcpyDeviceToHost);

    // Show results
    // print_matrix(h_c, "h_c", MATRIX_WIDTH);

    // Free device memmory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}