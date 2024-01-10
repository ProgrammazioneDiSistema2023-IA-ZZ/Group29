extern crate prost_build;

fn main() {
    // Add the `prost-build` dependency
    prost_build::compile_protos(&["src/onnx/onnx-data.proto3"], &["src/"]).unwrap();

}
