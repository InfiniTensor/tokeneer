#![deny(warnings)]

mod bpe;

use regex::Regex;
use std::collections::HashMap;

pub use bpe::Bpe;

/// `utok` for token id.
#[allow(non_camel_case_types)]
pub type utok = u32;

pub trait Method {
    fn unk_token(&self) -> utok;
    fn vocab_size(&self) -> usize;
    fn internal_special(&self) -> impl IntoIterator<Item = (&str, utok)>;
    fn encode(&self, text: &str) -> impl IntoIterator<Item = utok> + '_;
    fn decode(&self, token: utok) -> &[u8];
}

pub struct Tokeneer<M> {
    method: M,
    special: HashMap<String, Vec<utok>>,
    special_regex: regex::Regex,
}

impl<M: Method> Tokeneer<M> {
    pub fn new(method: M) -> Self {
        let special = method
            .internal_special()
            .into_iter()
            .map(|(k, v)| (k.to_string(), vec![v]))
            .collect::<HashMap<_, _>>();
        let special_regex = build_pattern(special.keys());
        Self {
            method,
            special,
            special_regex,
        }
    }

    pub fn extend_special(&mut self, patterns: impl IntoIterator<Item = (String, Vec<utok>)>) {
        use std::collections::hash_map::Entry::{Occupied, Vacant};
        let mut any = false;
        for (k, v) in patterns {
            match self.special.entry(k) {
                Occupied(entry) => {
                    assert_eq!(entry.get(), &v);
                }
                Vacant(entry) => {
                    entry.insert(v);
                    any = true;
                }
            }
        }
        if any {
            self.special_regex = build_pattern(self.special.keys());
        }
    }

    pub fn encode(&self, text: &str) -> Vec<utok> {
        let mut ans = Vec::new();
        let mut start = 0;
        for m in self.special_regex.find_iter(text) {
            ans.extend(self.method.encode(&text[start..m.start()]));
            ans.extend_from_slice(&self.special[m.as_str()]);
            start = m.end();
        }
        ans.extend(self.method.encode(&text[start..]));
        ans
    }

    pub fn decode(&self, tokens: &[utok]) -> String {
        let mut ans = Vec::new();
        for &t in tokens {
            ans.extend_from_slice(self.method.decode(t));
        }
        String::from_utf8(ans).unwrap()
    }
    #[inline]
    pub fn internal(&self) -> &M {
        &self.method
    }
}

fn build_pattern<T: AsRef<str>>(text: impl IntoIterator<Item = T>) -> Regex {
    let mut pattern = String::new();
    let mut iter = text.into_iter();
    if let Some(p) = iter.next() {
        pattern.push_str(p.as_ref());
    }
    for p in iter {
        pattern.push('|');
        pattern.push_str(p.as_ref());
    }
    regex::Regex::new(&pattern).unwrap()
}

const fn as_byte_token(piece: &[u8]) -> Option<u8> {
    // 按结构分解并转换
    match piece {
        &[b'<', b'0', b'x', a, b, b'>'] if a.is_ascii_hexdigit() && b.is_ascii_hexdigit() => {
            // ascii 转数字
            #[inline(always)]
            const fn to_num(c: u8) -> u8 {
                match c {
                    b'0'..=b'9' => c - b'0',
                    b'a'..=b'f' => c - b'a' + 10,
                    b'A'..=b'F' => c - b'A' + 10,
                    _ => unreachable!(),
                }
            }

            Some(to_num(a) * 16 + to_num(b))
        }
        _ => None,
    }
}
