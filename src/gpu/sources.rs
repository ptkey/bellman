static DEFS_SRC : &str = include_str!("common/defs.cl");
static FR_SRC : &str = include_str!("fft/Fr.cl");
static FIELD_SRC : &str = include_str!("common/field.cl");
static FFT_SRC : &str = include_str!("fft/fft.cl");

static FQ_SRC : &str = include_str!("multiexp/Fq.cl");
static FIELD2_SRC : &str = include_str!("multiexp/field2.cl");
static EC_SRC : &str = include_str!("multiexp/ec.cl");
static MULTIEXP_SRC : &str = include_str!("multiexp/multiexp.cl");

fn field(name: &str) -> String {
    return String::from(FIELD_SRC)
        .replace("FIELD", name);
}

fn field2(field2: &str, field: &str) -> String {
    return String::from(FIELD2_SRC)
        .replace("FIELD2", field2)
        .replace("FIELD", field);
}

fn fft(field: &str) -> String {
    return String::from(FFT_SRC)
        .replace("FIELD", field);
}

fn ec(field: &str, point: &str) -> String {
    return String::from(EC_SRC)
        .replace("FIELD", field)
        .replace("POINT", point);
}

fn multiexp(point: &str) -> String {
    return String::from(MULTIEXP_SRC)
        .replace("POINT", point);
}

pub fn fft_kernel() -> String {
    return String::from(format!("{}\n{}\n{}\n{}",
        DEFS_SRC, FR_SRC,
        field("Fr"), fft("Fr")));
}

pub fn multiexp_kernel() -> String {
    return String::from(format!("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        DEFS_SRC, FQ_SRC,
        field("Fq"), ec("Fq", "G1"), multiexp("G1"),
        field2("Fq2", "Fq"), ec("Fq2", "G2"), multiexp("G2")));
}
