#![allow(warnings)]
use rand::Rng;
use std::collections::{HashSet,HashMap};
use gpt_dev_rust::rustpp::*;
use tch::nn::{Module, OptimizerConfig,ModuleT};
use tch::{Kind, nn, Device, Tensor};

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


fn get_batch(data:&Tensor,batch_size:&i64,block_size:&i64){
    let data_len = data.size()[0];
    let mut rng = rand::thread_rng();
    let ix: Vec<i64> = (0..batch_size.clone())
        .map(|_| rng.gen_range(0..(data_len - block_size.clone()) as i64))
        .collect();
    let mut x = Vec::<Tensor>::new();
    for &i in ix.iter() {
        let data_slice = data.narrow(0, i, block_size.clone());
        x.push(data_slice);
    }
    let mut y = Vec::<Tensor>::new();
    for &i in ix.iter() {
        let data_slice = data.narrow(0, i, block_size.clone());
        y.push(data_slice);
    }
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

#[derive(Debug)]
struct Head {
    key: nn::Linear,
    query: nn::Linear,
    value: nn::Linear,
    tril: Tensor,
    dropout: f64,
}

impl Head {
    fn new(vs: &nn::Path, n_embd: i64, head_size: i64, block_size: i64, dropout: f64) -> Head {
        Head {
            key: nn::linear(vs, n_embd, head_size, Default::default()),
            query: nn::linear(vs, n_embd, head_size, Default::default()),
            value: nn::linear(vs, n_embd, head_size, Default::default()),
            tril: Tensor::from(tch::Tensor::tril(&Tensor::ones(&[block_size, block_size], (Kind::Float, tch::Device::Cpu)),0)),
            dropout,
        }
    }
}

impl Module for Head{
        fn forward(&self, x: &Tensor) -> Tensor {
        let (b, t, c) = x.size3().unwrap();
        let k = self.key.forward(x);   // (B, T, C)
        let q = self.query.forward(x); // (B, T, C)

        // Compute attention scores ("affinities")
        let wei = q.matmul(&k.transpose(-1, -2)) * (c as f64).powf(-0.5); // (B, T, C) @ (B, C, T) -> (B, T, T)

        // Set lower triangular elements to -inf to implement causality (optional)
        let triu_mask = self.tril.slice(1, 0, t, 1).unsqueeze(0); // (1, T, T)
        let causal_mask = triu_mask.full_like(f64::NEG_INFINITY);
        let wei = wei * causal_mask;

        // Apply softmax to get attention weights
        let wei = wei.softmax(-1,Kind::Float);

        // Apply dropout to attention weights (optional)
        let wei = if self.dropout > 0.0 {
            wei.dropout(self.dropout, true)
        } else {
            wei
        };

        // Perform the weighted aggregation of the values
        let v = self.value.forward(x); // (B, T, C)
        let out = wei.matmul(&v); // (B, T, T) @ (B, T, C) -> (B, T, C)

        out
    }
}
