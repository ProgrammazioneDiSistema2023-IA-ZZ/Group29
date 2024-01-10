use prost::Message;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write, Error};


pub fn write<T: Message>(object: &T, path: &str) -> Result<(), Error> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let buffer = object.encode_to_vec();
    writer.write_all(&buffer)
}

pub fn read<T: Message + Default>(path: &str) -> Result<T, Error> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;
    Ok(T::decode(buffer.as_slice())?)
}