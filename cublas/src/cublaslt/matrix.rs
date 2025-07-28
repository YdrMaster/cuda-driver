use crate::bindings::{hcblasLtMatrixLayout_t, hcblasLtOrder_t, hpccDataType_t};
use cuda::AsRaw;
use std::ptr::null_mut;

#[repr(transparent)]
pub struct CublasLtMatrix(hcblasLtMatrixLayout_t);

unsafe impl Send for CublasLtMatrix {}
unsafe impl Sync for CublasLtMatrix {}

impl AsRaw for CublasLtMatrix {
    type Raw = hcblasLtMatrixLayout_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Drop for CublasLtMatrix {
    #[inline]
    fn drop(&mut self) {
        cublas!(hcblasLtMatrixLayoutDestroy(self.0));
    }
}

impl From<CublasLtMatrixLayout> for CublasLtMatrix {
    fn from(layout: CublasLtMatrixLayout) -> Self {
        let mut matrix = null_mut();
        cublas!(hcblasLtMatrixLayoutCreate(
            &mut matrix,
            layout.data_type,
            layout.rows,
            layout.cols,
            layout.major_stride,
        ));
        let order = unsafe { layout.order.as_raw() };
        cublas!(hcblasLtMatrixLayoutSetAttribute(
            matrix,
            hcblasLtMatrixLayoutAttribute_t::HCBLASLT_MATRIX_LAYOUT_ORDER,
            (&raw const order).cast(),
            size_of_val(&order),
        ));
        cublas!(hcblasLtMatrixLayoutSetAttribute(
            matrix,
            hcblasLtMatrixLayoutAttribute_t::HCBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            (&raw const layout.batch).cast(),
            size_of_val(&layout.batch),
        ));
        cublas!(hcblasLtMatrixLayoutSetAttribute(
            matrix,
            hcblasLtMatrixLayoutAttribute_t::HCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            (&raw const layout.stride).cast(),
            size_of_val(&layout.stride),
        ));
        Self(matrix)
    }
}

pub struct CublasLtMatrixLayout {
    pub data_type: hpccDataType_t,
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
    type Raw = hcblasLtOrder_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        match self {
            Self::RowMajor => Self::Raw::HCBLASLT_ORDER_ROW,
            Self::ColMajor => Self::Raw::HCBLASLT_ORDER_COL,
        }
    }
}

#[test]
fn test_behavior() {
    // 据测试，这个类型是 CPU 上的，不需要在上下文中调用。
    let _mat = CublasLtMatrix::from(CublasLtMatrixLayout {
        data_type: hpccDataType_t::HPCC_R_16F,
        batch: 1,
        stride: 0,
        rows: 128,
        cols: 192,
        major_stride: 128,
        order: MatrixOrder::RowMajor,
    });
}
