extern crate ocl;

use self::ocl::ProQue;

// Currently it just doubles the input elements
pub fn fft(a: &mut [i32]) -> ocl::Result<()> {
    let src = r#"
        __kernel void add(__global int* buffer) {
            buffer[get_global_id(0)] *= 2;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims(a.len())
        .build()?;

    let buffer = pro_que.create_buffer::<i32>()?;

    let mut vec = vec![0i32; buffer.len()];
    for i in 0..a.len() { vec[i] = a[i]; }

    buffer.write(&vec).enq()?;

    let kernel = pro_que.kernel_builder("add")
        .arg(&buffer)
        .build()?;

    unsafe { kernel.enq()?; }

    buffer.read(&mut vec).enq()?;

    for i in 0..a.len() { a[i] = vec[i]; }
    Ok(())
}
