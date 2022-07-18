#ifndef __SimpleMatrix_h__
#define __SimpleMatrix_h__

#include "DeviceMatrixBase.h"
#include <array>

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

  __host__ __device__ inline T &operator()(SizeType index) {
    static_assert(Row == 1 || Col == 1);
    assert(index < Row && index < Col);
    return (*this)[index];
  }

  __host__ __device__ inline const T &operator()(SizeType index) const {
    static_assert(Row == 1 || Col == 1);
    assert(index < Row && index < Col);
    return (*this)[index];
  }
};

template <typename T, SizeType N, bool RowMajor = false>
using SimpleVector = SimpleMatrix<T, N, 1, RowMajor>;

#endif
