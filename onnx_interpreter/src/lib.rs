pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod file;

pub mod operations;

pub mod interpreter;

pub mod utils;

pub mod array;

pub mod attribute;
