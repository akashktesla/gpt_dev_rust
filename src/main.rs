#![allow(warnings)]
use std::collections::HashSet;
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
    
    //encoding
    
        
    



}
