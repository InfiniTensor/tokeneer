use std::{collections::HashMap, sync::LazyLock};

static BYTE_TO_UTF8: LazyLock<HashMap<u8, char>> = LazyLock::new(unicode_byte_to_utf8_map);
static UTF8_TO_BYTE: LazyLock<HashMap<char, u8>> = LazyLock::new(unicode_utf8_to_byte_map);

/// 创建一个从字节到 UTF-8 字符串的映射,主要用于gpt2
fn unicode_byte_to_utf8_map() -> HashMap<u8, char> {
    let mut map = HashMap::new();

    // 映射 ASCII 可打印字符 '!' 到 '~'
    for ch in 0x21..=0x7E {
        map.insert(ch as u8, char::from_u32(ch).unwrap());
    }

    // 映射拉丁字符 '¡' 到 '¬'
    for ch in 0xA1..=0xAC {
        map.insert(ch as u8, char::from_u32(ch).unwrap());
    }

    // 映射拉丁字符 '®' 到 'ÿ'
    for ch in 0xAE..=0xFF {
        map.insert(ch as u8, char::from_u32(ch).unwrap());
    }

    // 为剩余的字节值分配映射
    let mut n = 0;
    for ch in 0..256 {
        if let std::collections::hash_map::Entry::Vacant(e) = map.entry(ch as u8) {
            e.insert(char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }
    map
}

/// 创建一个从字节到 UTF-8 字符串的映射，主要用于gpt2
fn unicode_utf8_to_byte_map() -> HashMap<char, u8> {
    let mut map = HashMap::new();

    // 映射 ASCII 可打印字符 '!' 到 '~'
    for ch in 0x21..=0x7E {
        map.insert(char::from_u32(ch).unwrap(), ch as u8);
    }

    // 映射拉丁字符 '¡' 到 '¬'
    for ch in 0xA1..=0xAC {
        map.insert(char::from_u32(ch).unwrap(), ch as u8);
    }

    // 映射拉丁字符 '®' 到 'ÿ'
    for ch in 0xAE..=0xFF {
        map.insert(char::from_u32(ch).unwrap(), ch as u8);
    }

    // 为剩余的字节值分配映射
    let mut n = 0;
    for ch in 0..256 {
        if !map.contains_key(&char::from_u32(ch).unwrap()) {
            map.insert(char::from_u32(256 + n).unwrap(), ch as u8);
            n += 1;
        }
    }
    map
}

pub fn unicode_byte_to_utf8(byte: u8) -> char {
    *BYTE_TO_UTF8.get(&byte).unwrap()
}

pub fn unicode_utf8_to_byte(utf8: char) -> u8 {
    *UTF8_TO_BYTE.get(&utf8).unwrap()
}

pub fn llama_decode_text(text: &str) -> Vec<u8> {
    text.chars().map(unicode_utf8_to_byte).collect()
}

#[test]
fn convert() {
    let s = String::from("你好");
    let p: String = s
        .into_bytes()
        .iter()
        .map(|s| unicode_byte_to_utf8(*s))
        .collect();
    print!("dsf {p:?}");
    assert!(p == "ä½łå¥½");
    let string =
        unsafe { String::from_utf8_unchecked(vec![196, 160, 195, 165, 194, 165, 196, 169]) };
    let a = llama_decode_text(&string);
    println!("dsf {a:?}");
    // println!("ds {:?}",b.get(&'▁').unwrap());
}

#[cfg(test)]
mod test {
    use crate::{Bpe, TextBuf};
    use ggus::GGuf;
    use std::fs::File;

    #[test]
    fn bpe_from_gguf() {
        println!("1");
        let gguf =
            File::open("/home/yangderui/repos/DeepSeek-R1-Distill-Qwen-32B-v0.0-F16.gguf").unwrap();
        println!("2");
        let gguf = unsafe { memmap2::Mmap::map(&gguf) }.unwrap();
        println!("3");
        let gguf = GGuf::new(&gguf).unwrap();
        println!("4");
        let tokeneer = Bpe::from_gguf(&gguf);
        println!("5");
        let mut buf = TextBuf::new();
        let ans = tokeneer.decode(&[4891, 98, 229], &mut buf);
        println!("{ans} {:?}", ans.as_bytes());
        println!("6");
    }
}
