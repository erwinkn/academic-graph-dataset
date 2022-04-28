use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, Write},
    path::Path,
};

use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct AbstractInfo {
    IndexLength: usize,

    InvertedIndex: HashMap<String, Vec<usize>>,
}

fn main() {
    // The path is from the root directory, where Cargo.toml lives
    let input_path = Path::new("./data/abstracts.txt");
    let output_path = Path::new("./processed/abstracts.txt");
    let input_file = File::open(input_path).unwrap();
    let mut output_file = File::create(output_path).unwrap();

    let reader = BufReader::new(input_file);
    let mut counter = 0;
    for line in reader.lines() {
        let line = line.unwrap();
        process_line(line, &mut output_file);
        counter += 1;
    }
    println!("Finished parsing {} lines!", counter);
}

fn process_line(line: String, output: &mut File) {
    let mut pos = 0;
    // Parse the ID manually
    let chars = line.chars();
    for c in chars {
        if !c.is_ascii_digit() {
            break;
        }
        pos += 1;
    }
    let id = &line[0..pos];

    // Skip the 4 dashes
    pos += 4;

    // The rest is a regular JSON dictionary
    let info: AbstractInfo = serde_json::from_str(&line[pos..]).unwrap();

    let mut reordered: Vec<&str> = vec![""; info.IndexLength];

    for (token, occurences) in &info.InvertedIndex {
        for &idx in occurences {
            reordered[idx] = token;
        }
    }
    
    // Write everything manually, to rectify some of the tokens
    output.write(id.as_bytes()).unwrap();
    output.write(",\"".as_bytes()).unwrap();

    for token in reordered {
        // Some tokens contain newlines and carriage returns...
        // We can just have all tokens separated by whitespaces, since the data will be tokenized again before ingestion by BERT
        for word in token.split_whitespace() {
            output.write(word.as_bytes()).unwrap();
            output.write(&[b' ']).unwrap();
        }
    }
    output.write("\"\n".as_bytes()).unwrap();
}
