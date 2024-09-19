use crate::{utok, Method};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
    slice::from_ref,
    sync::LazyLock,
};

pub struct Tokeneer<M> {
    method: M,
    special: HashMap<String, TokenSeq>,
    special_regex: Regex,
}

enum TokenSeq {
    Single(utok),
    Multi(Box<[utok]>),
}

impl Deref for TokenSeq {
    type Target = [utok];
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Single(t) => from_ref(t),
            Self::Multi(t) => t,
        }
    }
}

impl<M: Method> Tokeneer<M> {
    pub fn new(method: M) -> Self {
        let special = method
            .internal_special()
            .into_iter()
            .filter(|(k, _)| k.is_ascii())
            .map(|(k, v)| (k.to_string(), TokenSeq::Single(v)))
            .collect::<HashMap<_, _>>();
        let special_regex = build_pattern(special.keys());
        Self {
            method,
            special,
            special_regex,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<utok> {
        let mut ans = Vec::new();
        let mut start = 0;
        if !self.special_regex.as_str().is_empty() {
            for m in self.special_regex.find_iter(text) {
                ans.extend(self.method.encode(&text[start..m.start()]));
                ans.extend_from_slice(&self.special[m.as_str()]);
                start = m.end();
            }
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
}

impl<M> Tokeneer<M> {
    pub fn extend_special(&mut self, patterns: impl IntoIterator<Item = (String, Vec<utok>)>) {
        use std::collections::hash_map::Entry::{Occupied, Vacant};
        let mut any = false;
        for (k, v) in patterns {
            match self.special.entry(k) {
                Occupied(entry) => {
                    assert_eq!(&**entry.get(), &v);
                }
                Vacant(entry) => {
                    entry.insert(TokenSeq::Multi(v.into_boxed_slice()));
                    any = true;
                }
            }
        }
        if any {
            self.special_regex = build_pattern(self.special.keys());
        }
    }

    #[inline]
    pub fn internal(&self) -> &M {
        &self.method
    }
}

fn build_pattern<'a>(text: impl IntoIterator<Item = &'a String>) -> Regex {
    static SPECIAL: LazyLock<HashSet<char>> = LazyLock::new(|| {
        HashSet::from([
            '*', '.', '?', '+', '^', '$', '|', '/', '\\', '(', ')', '[', ']', '{', '}',
        ])
    });

    let mut pattern = String::new();
    for p in text {
        for c in p.chars() {
            if SPECIAL.contains(&c) {
                pattern.push('\\');
            }
            pattern.push(c);
        }
        pattern.push('|');
    }
    pattern.pop();

    Regex::new(&pattern).unwrap()
}
