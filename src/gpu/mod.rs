mod fft;
pub use self::fft::*;

pub fn find_gpu() -> bool {
    //use ocl::{Platform, Device, flags};
    /*let platforms = Platform::list();
    let mut test = false;
    println!("Looping through avaliable platforms ({}):", platforms.len());

    for p_idx in 0..platforms.len() {
        let platform = &platforms[p_idx];

        let devices = Device::list(platform, Some(flags::DEVICE_TYPE_GPU)).unwrap();

        if devices.is_empty() { continue; }

        test = true;
        // for device in devices.iter() {
        //     println!("Device Name: {:?}, Vendor: {:?}", device.name().unwrap(),
        //         device.vendor().unwrap());
        // }
    }*/
    true
}
