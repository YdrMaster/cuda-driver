use crate::{
    CurrentCtx, Stream,
    bindings::{CUgraph, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL as GLOBAL},
};
use context_spore::{AsRaw, impl_spore};
use std::{ffi::CString, marker::PhantomData, ops::Deref, path::Path, ptr::null_mut, str::FromStr};

impl_spore!(Graph and GraphSpore by (CurrentCtx, CUgraph));

#[repr(transparent)]
pub struct CaptureStream<'ctx>(Stream<'ctx>);

impl<'ctx> Stream<'ctx> {
    pub fn capture(self) -> CaptureStream<'ctx> {
        driver!(cuStreamBeginCapture_v2(self.as_raw(), GLOBAL));
        CaptureStream(self)
    }
}

impl<'ctx> CaptureStream<'ctx> {
    pub fn end(self) -> Graph<'ctx> {
        let mut graph = null_mut();
        driver!(cuStreamEndCapture(self.0.as_raw(), &mut graph));
        Graph(unsafe { self.0.ctx().wrap_raw(graph) }, PhantomData)
    }
}

impl<'ctx> Deref for CaptureStream<'ctx> {
    type Target = Stream<'ctx>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for Graph<'_> {
    #[inline]
    fn drop(&mut self) {
        driver!(cuGraphDestroy(self.0.rss));
    }
}

impl AsRaw for Graph<'_> {
    type Raw = CUgraph;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss
    }
}

impl Graph<'_> {
    pub fn save_dot(&self, path: impl AsRef<Path>) {
        let path = CString::from_str(&path.as_ref().display().to_string()).unwrap();
        driver!(cuGraphDebugDotPrint(
            self.as_raw(),
            path.as_ptr().cast(),
            u32::MAX
        ))
    }
}

#[test]
fn test_behavior() {
    use crate::{Device, Ptx, Symbol, params};

    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    let device = Device::new(0);
    device.context().apply(|ctx| {
        const CODE: &str = r#"
extern "C" __global__ void add(float *a, float const *b) {
  a[threadIdx.x] += b[threadIdx.x];
}
extern "C" __global__ void sub(float *a, float const *b) {
  a[threadIdx.x] -= b[threadIdx.x];
}
extern "C" __global__ void mul(float *a, float const *b) {
  a[threadIdx.x] *= b[threadIdx.x];
}
"#;

        let (ptx, _log) = Ptx::compile(CODE, device.compute_capability());
        let module = ctx.load(&ptx.unwrap());
        let mut kernels = Symbol::search(CODE).map(|name| module.get_kernel(name.to_c_string()));
        let add = kernels.next().unwrap();
        let sub = kernels.next().unwrap();
        let mul = kernels.next().unwrap();

        let a_host = ctx.malloc_host::<f32>(1024);
        let b_host = ctx.malloc_host::<f32>(1024);
        let c_host = ctx.malloc_host::<f32>(1024);
        let d_host = ctx.malloc_host::<f32>(1024);
        let mut ans = ctx.malloc_host::<f32>(1024);

        let stream = ctx.stream();
        let stream = stream.capture();

        let mut a = stream.from_host(&a_host);
        let mut b = stream.from_host(&b_host);
        let mut c = stream.from_host(&c_host);
        let mut d = stream.from_host(&d_host);

        add.launch(
            1,
            1024,
            params![a.as_mut_ptr(), b.as_mut_ptr()].as_ptr(),
            0,
            Some(&stream),
        );
        sub.launch(
            1,
            1024,
            params![c.as_mut_ptr(), d.as_mut_ptr()].as_ptr(),
            0,
            Some(&stream),
        );
        mul.launch(
            1,
            1024,
            params![a.as_mut_ptr(), c.as_mut_ptr()].as_ptr(),
            0,
            Some(&stream),
        );
        driver!(cuMemcpyDtoHAsync_v2(
            ans.as_mut_ptr().cast(),
            a.as_ptr() as _,
            4096,
            stream.as_raw()
        ));

        a.drop_on(&stream);
        b.drop_on(&stream);
        c.drop_on(&stream);
        d.drop_on(&stream);

        let graph = stream.end();
        graph.save_dot(std::env::current_dir().unwrap().join("graph.dot"))
    })
}
