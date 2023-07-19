use std::fs::File;
use std::fs::write;
use std::fs::create_dir;
use std::io::prelude::*;

pub fn create_file(path:String,file_name:String){
    match create_dir(&path){
        Ok(_) => (),
        Err(_) => () ,
    }
    match File::create(format!("{}/{}",path,file_name)){
        Ok(_) => (),
        Err(_) => () ,
    }
}//end

pub fn read_file(path:&String)->String{
    let mut _file = File::open(path).expect("can't open file");
    let mut contents = String::new();
    _file.read_to_string(&mut contents).expect("can't read the file");
    contents = format!(r#"{}"#,contents); 
    return contents
}

pub fn write_file(path:&String,contents:&String){
    match write(path,contents){
        Ok(_) =>(),
        Err(_) => () ,
    }
}
