use crate::bindings::{macaDataType_t, mcblasLtMatrixLayout_t, mcblasLtOrder_t};
use cuda::AsRaw;
use std::ptr::null_mut;

#[repr(transparent)]
pub struct CublasLtMatrix(mcblasLtMatrixLayout_t);

unsafe impl Send for CublasLtMatrix {}
unsafe impl Sync for CublasLtMatrix {}

impl AsRaw for CublasLtMatrix {
    type Raw = mcblasLtMatrixLayout_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Drop for CublasLtMatrix {
    #[inline]
    fn drop(&mut self) {
        cublas!(mcblasLtMatrixLayoutDestroy(self.0));
    }
}

impl From<CublasLtMatrixLayout> for CublasLtMatrix {
    fn from(layout: CublasLtMatrixLayout) -> Self {
        let mut matrix = null_mut();
        cublas!(mcblasLtMatrixLayoutCreate(
            &mut matrix,
            layout.data_type,
            layout.rows,
            layout.cols,
            layout.major_stride,
        ));
        let order = unsafe { layout.order.as_raw() };
        cublas!(mcblasLtMatrixLayoutSetAttribute(
            matrix,
            mcblasLtMatrixLayoutAttribute_t::MCBLASLT_MATRIX_LAYOUT_ORDER,
            (&raw const order).cast(),
            size_of_val(&order),
        ));
        cublas!(mcblasLtMatrixLayoutSetAttribute(
            matrix,
            mcblasLtMatrixLayoutAttribute_t::MCBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            (&raw const layout.batch).cast(),
            size_of_val(&layout.batch),
        ));
        cublas!(mcblasLtMatrixLayoutSetAttribute(
            matrix,
            mcblasLtMatrixLayoutAttribute_t::MCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            (&raw const layout.stride).cast(),
            size_of_val(&layout.stride),
        ));
        Self(matrix)
    }
}

pub struct CublasLtMatrixLayout {
    pub data_type: macaDataType_t,
    pub batch: i32,
    pub stride: i64,
    pub rows: u64,
    pub cols: u64,
    pub major_stride: i64,
    pub order: MatrixOrder,
}

#[derive(Clone, Copy, Debug)]
pub enum MatrixOrder {
    RowMajor,
    ColMajor,
}

impl AsRaw for MatrixOrder {
    type Raw = mcblasLtOrder_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        match self {
            Self::RowMajor => Self::Raw::MCBLASLT_ORDER_ROW,
            Self::ColMajor => Self::Raw::MCBLASLT_ORDER_COL,
        }
    }
}

#[test]
fn test_behavior() {
    // 据测试，这个类型是 CPU 上的，不需要在上下文中调用。
    let _mat = CublasLtMatrix::from(CublasLtMatrixLayout {
        data_type: macaDataType_t::MACA_R_16F,
        batch: 1,
        stride: 0,
        rows: 128,
        cols: 192,
        major_stride: 128,
        order: MatrixOrder::RowMajor,
    });
}
