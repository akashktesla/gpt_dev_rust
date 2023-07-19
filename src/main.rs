#![allow(warnings)]
use std::collections::{HashSet,HashMap};
use gpt_dev_rust::rustpp::*;
use tch::nn::{Module, OptimizerConfig};
use tch::{kind, nn, Device, Tensor};

fn main(){
    let batch_size = 16; // how many independent sequences will we process in parallel?
    let block_size = 32; // what is the maximum context length for predictions?
    let max_iters = 5000;
    let eval_interval = 100;
    let learning_rate = 1e-3;
    let eval_iters = 200;
    let n_embd = 64;
    let n_head = 4;
    let n_layer = 4;
    let dropout = 0.0;

    let text = read_file(&String::from("geng.txt"));
    let chars = text.chars().collect::<HashSet<char>>().into_iter().collect::<Vec<char>>();
    let vocab_size = chars.len();
    
    //for encoding
    // c(char) to i(index)
    let ctoi:HashMap<char,i32> = chars
        .iter()
        .enumerate()
        .map(|(i, &ch)|{ (ch, i as i32)})
        .collect();
    // for decoding
    // i(index) to c(char)
    let itoc:HashMap<i32,char> = chars
        .iter()
        .enumerate()
        .map(|(i,&ch)|{(i as i32,ch)})
        .collect();
    
    let encoded_data = encode(&text,&ctoi);
    let data = Tensor::from_slice(&encoded_data).to_kind(tch::Kind::Int64);
    let data_len = data.size()[0];
    let n = (0.9*data_len as f32) as i64; // first 90% will be train, rest val
    let mut split_data = data.split(n, 0);
    let val_data = split_data.pop().unwrap(); 
    let train_data = split_data.pop().unwrap(); 
    println!("train data: {:?}",train_data);
    train_data.print();
    println!("test data: {:?}",val_data);
        
}


fn get_batch(data:&Tensor){
    let len = data.size()[0];

}


fn encode(str:&String,ctoi:&HashMap<char,i32>)->Vec<i32>{ 
    let mut returns = Vec::new();
    for i in str.chars(){
        returns.push(ctoi.get(&i).unwrap().clone());
    }
    return returns;
}

fn decode(vec:Vec<i32>,itoc:HashMap<i32,char>)->String{
    let mut returns = String::new();
    for i in vec{
        returns.push(itoc.get(&i).unwrap().clone());
    }
    return returns;
}







