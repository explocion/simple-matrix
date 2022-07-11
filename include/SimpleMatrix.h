#ifndef __SimpleMatrix_h__
#define __SimpleMatrix_h__

#include <array>
#include <cassert>
#include <cstdlib>
#include <type_traits>

// I have already filed an issue to llvm/clang for using static_assert in shared
// variable constructor. In the future it is possible to just use eigen matrix
// in device code. Therefore, only index, context, and evaluators are needed

// Interface for changing SizeType
#ifndef SIZE_TYPE
#define SIZE_TYPE std::size_t
#endif

using SizeType = SIZE_TYPE;

// As my recent interest in Rust, I would like to make clear what objects are
// mutable and immutable. MatrixIndex, DeviceMatrixContext ought to be
// immutable.

template <typename T, SizeType Row, SizeType Col, bool RowMajor = false>
class SimpleMatrix : public std::array<T, Row * Col> {
public:
  using Self = SimpleMatrix<T, Row, Col>;
  using ElementType = T;
  static constexpr auto Rows = Row;
  static constexpr auto Cols = Col;
  static constexpr auto RowMajorOverride = RowMajor;

  __host__ __device__ static inline SizeType serializeIndex(SizeType row,
                                                            SizeType col) {
    assert(row < Row && col < Col);
    if constexpr (RowMajorOverride) {
      return row * Col + col;
    } else {
      return col * Row + row;
    }
  }

  __host__ __device__ inline T &operator()(SizeType row, SizeType col = 0) {
    assert(row < Row && col < Col);
    return (*this)[serializeIndex(row, col)];
  }

  __host__ __device__ inline const T &operator()(SizeType row,
                                                 SizeType col = 0) const {
    assert(row < Row && col < Col);
    return (*this)[serializeIndex(row, col)];
  }
};

template <typename T, SizeType N, bool RowMajor = false>
class SimpleVector : public SimpleMatrix<T, N, 1, RowMajor> {
public:
  using Self = SimpleVector<T, N, RowMajor>;
  using Base = SimpleMatrix<T, N, 1>;
  using ElementType = T;
  static constexpr auto Rows = N;
  static constexpr auto Cols = 1;
  static constexpr auto RowMajorOverride = RowMajor;

  __host__ __device__ inline T &operator()(SizeType index) {
    assert(index < N);
    return (*this)[index];
  }

  __host__ __device__ inline const T &operator()(SizeType index) const {
    assert(index < N);
    return (*this)[index];
  }
};

#endif
