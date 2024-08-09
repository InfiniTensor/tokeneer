//! l-p-e for Longest Prefix Encoding

use crate::{
    utok,
    vocab::{CollectedVocab, CompressedVocab},
    Method,
};
use patricia_tree::PatriciaMap;
use std::{collections::HashSet, pin::Pin};

pub struct Lpe {
    /// 保存所有词的字符串内容，以 u8 为单位所以不需要对齐，占用空间少
    vocabs: Pin<Box<[u8]>>,
    /// 按 token 顺序保存元信息
    tokens: Box<[(u32, u32)]>,
    /// 词汇的前缀树
    trie: PatriciaMap<utok>,
    /// 用于索引单字节 token，因此不需要其他元信息
    bytes: Box<[utok; 256]>,
    /// token: <unk>
    unk: utok,
}

impl Lpe {
    pub fn from_vocabs_txt(txt: &[u8]) -> Self {
        Self::new(
            unsafe { std::str::from_utf8_unchecked(txt) }
                .lines()
                .map(|line| {
                    line.strip_prefix('"')
                        .unwrap()
                        .strip_suffix('"')
                        .unwrap()
                        .as_bytes()
                }),
            0,
        )
    }

    pub fn new<'a>(vocabs: impl IntoIterator<Item = &'a [u8]>, unk: utok) -> Self {
        let CollectedVocab {
            vocabs,
            total_len,
            bytes,
        } = CollectedVocab::collect(vocabs, unk);
        let CompressedVocab { vocabs, slices } = CompressedVocab::new(&vocabs, total_len);
        let tokens = slices
            .into_iter()
            .map(|(off, len)| (off as u32, len as u32))
            .collect::<Box<_>>();

        let bytes_set = bytes.iter().chain(&[unk]).cloned().collect::<HashSet<_>>();
        let trie = tokens
            .iter()
            .enumerate()
            .filter(|&(i, _)| !bytes_set.contains(&(i as utok)))
            .map(|(i, &(off, len))| (&vocabs[off as usize..][..len as usize], i as utok))
            .collect();

        // println!(
        //     "Building LPE vocab, detected {} tokens, compressed to {} bytes from {total_len} bytes",
        //     tokens.len(),
        //     vocabs.len(),
        // );

        Self {
            vocabs,
            tokens,
            trie,
            bytes,
            unk,
        }
    }

    /// token id -> token meta
    #[inline(always)]
    fn token(&self, token: utok) -> &[u8] {
        let (off, len) = self.tokens[token as usize];
        &self.vocabs[off as usize..][..len as usize]
    }
}

impl Method for Lpe {
    #[inline]
    fn unk_token(&self) -> utok {
        self.unk
    }
    #[inline]
    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
    #[inline]
    fn internal_special(&self) -> impl IntoIterator<Item = (&str, utok)> {
        []
    }
    #[inline]
    fn encode(&self, text: &str) -> impl IntoIterator<Item = utok> + '_ {
        let mut text = text.as_bytes();
        let mut tokens = Vec::<utok>::new();

        while !text.is_empty() {
            let (tok, len) = match self.trie.get_longest_common_prefix(text) {
                Some((pre, tok)) => (*tok, pre.len()),
                None => (self.bytes[text[0] as usize], 1),
            };
            tokens.push(tok);
            text = &text[len..];
        }

        tokens
    }
    #[inline]
    fn decode(&self, token: utok) -> &[u8] {
        self.token(token)
    }
}
