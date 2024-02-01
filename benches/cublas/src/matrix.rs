use crate::bindings::{self as cu, cublas};
use cuda::AsRaw;
use std::ptr::null_mut;

#[repr(transparent)]
pub struct CublasLtMatrix(cu::cublasLtMatrixLayout_t);

impl AsRaw for CublasLtMatrix {
    type Raw = cu::cublasLtMatrixLayout_t;

    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Drop for CublasLtMatrix {
    #[inline]
    fn drop(&mut self) {
        cublas!(cublasLtMatrixLayoutDestroy(self.0));
    }
}

impl From<CublasLtMatrixLayout> for CublasLtMatrix {
    fn from(layout: CublasLtMatrixLayout) -> Self {
        let mut matrix = null_mut();
        cublas!(cublasLtMatrixLayoutCreate(
            &mut matrix,
            layout.data_type,
            layout.rows,
            layout.cols,
            layout.major_stride,
        ));
        let order = unsafe { layout.order.as_raw() };
        cublas!(cublasLtMatrixLayoutSetAttribute(
            matrix,
            cu::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
            &order as *const _ as _,
            std::mem::size_of_val(&order),
        ));
        cublas!(cublasLtMatrixLayoutSetAttribute(
            matrix,
            cu::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &layout.batch_count as *const _ as _,
            std::mem::size_of_val(&layout.batch_count),
        ));
        cublas!(cublasLtMatrixLayoutSetAttribute(
            matrix,
            cu::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &layout.batch_stride as *const _ as _,
            std::mem::size_of_val(&layout.batch_stride),
        ));
        Self(matrix)
    }
}

pub struct CublasLtMatrixLayout {
    pub data_type: cu::cudaDataType,
    pub rows: u64,
    pub cols: u64,
    pub major_stride: i64,
    pub order: MatrixOrder,
    pub batch_count: i32,
    pub batch_stride: i64,
}

impl Default for CublasLtMatrixLayout {
    fn default() -> Self {
        Self {
            data_type: cu::cudaDataType::CUDA_R_32F,
            rows: 1,
            cols: 1,
            major_stride: 1,
            order: MatrixOrder::ColMajor,
            batch_count: 1,
            batch_stride: 0,
        }
    }
}

pub enum MatrixOrder {
    RowMajor,
    ColMajor,
}

impl AsRaw for MatrixOrder {
    type Raw = cu::cublasLtOrder_t;

    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        match self {
            Self::RowMajor => Self::Raw::CUBLASLT_ORDER_ROW,
            Self::ColMajor => Self::Raw::CUBLASLT_ORDER_COL,
        }
    }
}
