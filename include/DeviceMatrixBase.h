#ifndef __DeviceMatrixBase_h__
#define __DeviceMatrixBase_h__

#include <cooperative_groups.h>
#ifdef CUDA_EIGEN_EXTEND
#include <Eigen/Dense>
#endif

// Interface for changing SizeType
#ifndef SIZE_TYPE
#define SIZE_TYPE std::size_t
#endif

using SizeType = SIZE_TYPE;

// As my recent interest in Rust, I would like to make clear what objects are
// mutable and immutable. MatrixIndex, DeviceMatrixContext ought to be
// immutable.

template <SizeType Row, SizeType Col, bool RowMajorOverride = false>
class MatrixIndex {
public:
  const SizeType row;
  const SizeType column;

  __host__ __device__ MatrixIndex(SizeType index)
      : row(RowMajorOverride ? (index / Col) : (index % Row)),
        column(RowMajorOverride ? (index % Col) : (index / Row)) {
    assert(index < Row * Col);
  }

  template <typename Derived>
  __host__ __device__ inline auto &elementOfMut(Derived &m) const {
    static_assert(Derived::RowsAtCompileTime >= Row ||
                      Derived::ColsAtCompileTime >= Col,
                  "Accessing Matrix with Larger Dimension is Unsafe.");
    return m(row, column);
  }

  template <typename Derived>
  __host__ __device__ inline auto elementOf(const Derived &m) const {
    static_assert(Derived::RowsAtCompileTime >= Row ||
                      Derived::ColsAtCompileTime >= Col,
                  "Accessing Matrix with Larger Dimension is Unsafe.");
    return m(row, column);
  }

#ifdef CUDA_EIGEN_EXTEND
  template <typename Derived>
  __host__ __device__ inline auto &
  elementOfMut(Eigen::MapBase<Derived> m) const {
    static_assert(Derived::RowsAtCompileTime >= Row ||
                      Derived::ColsAtCompileTime >= Col,
                  "Accessing Matrix with Larger Dimension is Unsafe.");
    return m(row, column);
  }

  template <typename Derived>
  __host__ __device__ inline auto elementOf(Eigen::MapBase<Derived> m) const {
    static_assert(Derived::RowsAtCompileTime >= Row ||
                      Derived::ColsAtCompileTime >= Col,
                  "Accessing Matrix with Larger Dimension is Unsafe.");
    return m(row, column);
  }

  template <typename Derived>
  __host__ __device__ inline auto
  elementOf(Eigen::MapBase<Derived, Eigen::ReadOnlyAccessors> m) const {
    static_assert(Derived::RowsAtCompileTime >= Row ||
                      Derived::ColsAtCompileTime >= Col,
                  "Accessing Matrix with Larger Dimension is Unsafe.");
    return m(row, column);
  }
#endif

  template <typename Derived, typename Expr>
  __device__ inline void evaluate(Derived &result,
                                  const Expr &expression) const {
    elementOfMut(result) = elementOf(expression);
  }

#ifdef CUDA_EIGEN_EXTEND
  template <typename Derived, typename Expr>
  __device__ inline void evaluate(Eigen::MapBase<Derived> result,
                                  const Expr &expression) const {
    elementOfMut(result) = elementOf(expression);
  }
#endif

  __host__ __device__ inline bool verify(void) const {
    return row < Row && column < Col;
  }
};

template <SizeType Row, SizeType Col, SizeType SizeOverride = Row *Col,
          SizeType IntendedLanes = 0>
class DeviceMatrixContext {
public:
  static constexpr SizeType Size = SizeOverride;
  using Group =
      cooperative_groups::thread_block_tile<Size,
                                            cooperative_groups::thread_block>;
  const Group group;
  const MatrixIndex<Row, Col> index;

  __host__ __device__ DeviceMatrixContext()
      : group(cooperative_groups::tiled_partition<Size>(
            cooperative_groups::this_thread_block())),
        index(group.thread_rank()) {
#ifndef NDEBUG
    assert((IntendedLanes == 0) ||
           (cooperative_groups::thread_block::size() / Size == IntendedLanes));
#endif
  }

  __device__ inline SizeType lane(void) const {
    return group.meta_group_rank();
  }

public:
  template <bool alias, typename Derived, typename Expr>
  __device__ inline void evaluate(Derived &result,
                                  const Expr &expression) const {
    if constexpr (alias) {
      auto temp = index.elementOf(expression);
      group.sync();
      index.elementOfMut(result) = temp;
    } else {
      index.evaluate(result, expression);
    }
  }

#ifdef CUDA_EIGEN_EXTEND
  template <bool alias, typename Derived, typename Expr>
  __device__ inline void evaluate(Eigen::MapBase<Derived> result,
                                  const Expr &expression) const {
    if constexpr (alias) {
      auto temp = index.elementOf(expression);
      group.sync();
      index.elementOfMut(result) = temp;
    } else {
      index.evaluate(result, expression);
    }
  }
#endif
};

#endif
