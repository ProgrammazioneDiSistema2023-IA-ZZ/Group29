use onnx_interpreter::onnx;
use onnx_interpreter::file;
use onnx_interpreter::onnx::ModelProto;

mod operations;

fn main() {
    //Example of write and read any object
    let path = "models/test.onnx";
    let mut x = onnx::ModelProto::default();
    x.producer_name = "test".to_string();

    //Write
    println!("Write object: \n{:?}\n", x);
    file::write(&x, path).unwrap();

    //Read
    let y = file::read::<ModelProto>(path).unwrap();
    println!("Read object: \n{:?}\n", y);    

    //Example of read linear regression file
    let path = "models/linear_regression.onnx";
    let model = file::read::<ModelProto>(path).unwrap();
    println!("Linear regression model: \n{:?}\n", model);

    //Operations
    operations::perform_operations(y);

}
