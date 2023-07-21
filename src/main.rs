#![allow(warnings)]
use lazy_static::lazy_static;
use lazy_static::lazy::Lazy;
use tch::{Kind, nn, Device, Tensor};
use tch::nn::{Module, OptimizerConfig,ModuleT,Linear};
use rand::Rng;
use std::collections::{HashSet,HashMap};
use std::default;
use gpt_dev_rust::rustpp::*;

static BATCH_SIZE:i64 = 16; // how many independent sequences will we process in parallel?
static BLOCK_SIZE:i64 = 32; // what is the maximum context length for predictions?
static MAX_ITERS:i64 = 5000;
static eval_interval:i64 = 100;
static LEARNING_RATE:f64 = 1e-3;
static EVAL_ITERS:i64 = 200;
static N_EMBD:i64 = 64;
static N_HEAD:i64 = 4;
static N_LAYER:i64 = 4;
static DROPOUT:f64 = 0.0;
lazy_static!{
    static ref VOCAB_SIZE:i64 = return_vocab_size();
}

fn return_vocab_size()->i64{
    let text = read_file(&String::from("geng.txt"));
    let chars = text.chars().collect::<HashSet<char>>().into_iter().collect::<Vec<char>>();
    return chars.len() as i64;
    }
fn main(){

    let text = read_file(&String::from("geng.txt"));
    let chars = text.chars().collect::<HashSet<char>>().into_iter().collect::<Vec<char>>();

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
    fn new(vs: &nn::Path, head_size: i64) -> Head {
        Head {
            key: nn::linear(vs, N_EMBD, head_size, Default::default()),
            query: nn::linear(vs, N_EMBD, head_size, Default::default()),
            value: nn::linear(vs, N_EMBD, head_size, Default::default()),
            tril: Tensor::from(tch::Tensor::tril(&Tensor::ones(&[BLOCK_SIZE, BLOCK_SIZE], (Kind::Float, tch::Device::Cpu)),0)),
            dropout: DROPOUT,
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

//         self.proj = nn.Linear(n_embd, n_embd)
//         self.dropout = nn.Dropout(dropout)
//
//     def forward(self, x):
//         out = torch.cat([h(x) for h in self.heads], dim=-1)
//         out = self.dropout(self.proj(out))
//         return out

#[derive(Debug)]
struct MultiHeadAttention{
    head: Vec<Head>,
    proj:Linear,
    dropout:f64,
}

impl MultiHeadAttention{
    fn new(vs:&nn::Path,num_heads:i32,head_size:i64)->MultiHeadAttention{
        let mut head = Vec::new();
        for _ in 0..num_heads{
            head.push(Head::new(vs,head_size))
        }

        return MultiHeadAttention { 
            head, 
            proj: nn::linear(vs, N_EMBD, N_EMBD,Default::default()),
            dropout: DROPOUT 
        };
    }
}
impl Module for MultiHeadAttention{
    fn forward(&self, x:&Tensor) -> Tensor {
        let mut out = tch::Tensor::cat(
            &self.head
            .iter()
            .map(|head| head.forward(&x))
            .collect::<Vec<_>>(),
            -1,
            );

        out = self.proj.forward(&out);
        return out;
    }
}


// class FeedFoward(nn.Module):
//     """ a simple linear layer followed by a non-linearity """

//     def __init__(self, n_embd):
//         super().__init__()
//         self.net = nn.Sequential(
//             nn.Linear(n_embd, 4 * n_embd),
//             nn.ReLU(),
//             nn.Linear(4 * n_embd, n_embd),
//             nn.Dropout(dropout),
//         )

//     def forward(self, x):
//         return self.net(x)

#[derive(Debug)]
struct FeedForward{
    net:nn::Sequential,
}
impl FeedForward{
    fn new(vs:&nn::Path)->FeedForward{
            let net = nn::seq()
                .add(nn::linear(vs,N_EMBD, 4 * N_EMBD, Default::default()))
                .add_fn(|xs|xs.relu())
                .add(nn::linear(vs,4 * N_EMBD, N_EMBD, Default::default()))
                .add_fn(|xs|xs.dropout(DROPOUT, true));
                
        return FeedForward { net };
    }
}
impl Module for FeedForward{
    fn forward(&self, xs: &Tensor) -> Tensor {
        return self.net.forward(xs);
    }
}

//     def __init__(self, n_embd, n_head):
//         # n_embd: embedding dimension, n_head: the number of heads we'd like
//         super().__init__()
//         head_size = n_embd // n_head
//         self.sa = MultiHeadAttention(n_head, head_size)
//         self.ffwd = FeedFoward(n_embd)
//         self.ln1 = nn.LayerNorm(n_embd)
//         self.ln2 = nn.LayerNorm(n_embd)

//     def forward(self, x):
//         x = x + self.sa(self.ln1(x))
//         x = x + self.ffwd(self.ln2(x))
//         return x

#[derive(Debug)]
struct Block{
    sa:MultiHeadAttention,
    ffwd:FeedForward,
    ln1:nn::LayerNorm,
    ln2:nn::LayerNorm,
}

impl Block{
    fn new(vs:&nn::Path)->Block{
        let head_size = N_EMBD/N_HEAD;
        return Block { 
            sa: MultiHeadAttention::new(vs,N_HEAD as i32, head_size),
            ffwd: FeedForward::new(&vs),
            ln1: nn::layer_norm(vs, vec![N_EMBD], Default::default()),
            ln2: nn::layer_norm(vs, vec![N_EMBD], Default::default())
        }
    }
}

impl Module for Block{
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut x = xs+self.sa.forward(&self.ln1.forward(xs));
        let temp = self.ffwd.forward(&self.ln2.forward(&x));
        x = x+temp;
        return x;
    }
}



// class BigramLanguageModel(nn.Module):

//     def __init__(self):
//         super().__init__()
//         # each token directly reads off the logits for the next token from a lookup table
//         self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
//         self.position_embedding_table = nn.Embedding(block_size, n_embd)
//         self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
//         self.ln_f = nn.LayerNorm(n_embd) # final layer norm
//         self.lm_head = nn.Linear(n_embd, vocab_size)

struct BigramLanguageModel{
    token_embedding_table :nn::Embedding,
    position_embedding_table: nn::Embedding,
    blocks: nn::Sequential,
    ln_f: nn::LayerNorm,
    lm_head: nn::Linear,
}

impl  BigramLanguageModel{
    fn new(vs:&nn::Path)->BigramLanguageModel{
        let mut blocks = nn::seq();
        for _ in 0..N_LAYER{
            blocks = blocks.add(Block::new(vs));
        }
        BigramLanguageModel { 
            token_embedding_table: nn::embedding(vs, *VOCAB_SIZE, N_EMBD, Default::default()),
            position_embedding_table: nn::embedding(vs, BLOCK_SIZE, N_EMBD, Default::default()),
            blocks,
            ln_f:nn::layer_norm(vs, vec![N_EMBD], Default::default()) ,
            lm_head: nn::linear(vs, N_EMBD, *VOCAB_SIZE, Default::default()) 
        }
    }
}




