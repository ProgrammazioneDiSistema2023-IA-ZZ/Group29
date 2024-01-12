pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod file;

pub mod operations;
