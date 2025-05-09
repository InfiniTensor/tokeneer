//! l-p-e for Longest Prefix Encoding

use crate::{
    Method, utok,
    vocab::{CollectedVocab, CompressedVocab, TokenType},
};
use patricia_tree::PatriciaMap;
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    pin::Pin,
    sync::LazyLock,
};

pub struct Lpe {
    /// 保存所有词的字符串内容，以 u8 为单位所以不需要对齐，占用空间少
    vocabs: Pin<Box<[u8]>>,
    /// 按 token 顺序保存元信息
    tokens: Box<[(u32, u32)]>,
    /// 词汇的前缀树
    trie: PatriciaMap<utok>,
    /// 用于索引单字节 token，因此不需要其他元信息
    bytes: Box<[utok; 256]>,
    /// 特殊词汇表
    special: Box<[utok]>,
    /// token: <unk>
    unk: utok,
}

impl Lpe {
    pub fn from_vocabs_txt(txt: &[u8]) -> Self {
        Self::from_collected_vocab(
            CollectedVocab::collect(
                unsafe { std::str::from_utf8_unchecked(txt) }
                    .lines()
                    .map(|line| {
                        line.strip_prefix('"')
                            .unwrap()
                            .strip_suffix('"')
                            .unwrap()
                            .as_bytes()
                    }),
                std::iter::repeat(TokenType::Normal),
                0,
            ),
            false,
        )
    }

    pub fn new<'a>(
        vocabs: impl IntoIterator<Item = &'a [u8]>,
        token_type: impl IntoIterator<Item = TokenType>,
        unk: utok,
        map_utf8: bool,
    ) -> Self {
        Self::from_collected_vocab(CollectedVocab::collect(vocabs, token_type, unk), map_utf8)
    }

    fn from_collected_vocab(vocab: CollectedVocab, map_utf8: bool) -> Self {
        let CollectedVocab {
            vocabs,
            total_len,
            bytes,
            special,
            unk,
            ..
        } = vocab;

        let CompressedVocab { vocabs, slices } = if map_utf8 {
            let vocabs = vocabs
                .into_iter()
                .enumerate()
                .map(|(i, token)| {
                    if special.contains(&(i as u32)) {
                        Cow::Borrowed(token)
                    } else {
                        let text = unsafe { std::str::from_utf8_unchecked(token) };
                        let mut utf8 = Vec::new();
                        for c in text.chars() {
                            let piece = [c].iter().collect::<String>();
                            if let Some(&c) = MAP_UTF8_TO_BYTE.get(&piece) {
                                utf8.push(c)
                            } else {
                                let c = c as u8;
                                utf8.extend_from_slice(format!("[UNK_BYTE_{c:#02x}]").as_bytes())
                            }
                        }
                        Cow::Owned(utf8)
                    }
                })
                .collect::<Vec<_>>();
            let vocabs = vocabs.iter().map(|s| &**s).collect::<Vec<_>>();
            CompressedVocab::new(&vocabs, total_len)
        } else {
            CompressedVocab::new(&vocabs, total_len)
        };

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
            special,
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
        self.special.iter().map(|&t| {
            let s = unsafe { std::str::from_utf8_unchecked(self.token(t)) };
            (s, t)
        })
    }

    fn encode(&self, text: &str) -> impl IntoIterator<Item = utok> + '_ {
        let mut text = text.as_bytes();
        let mut tokens = Vec::<utok>::new();

        while !text.is_empty() {
            let (tok, len) = match self.trie.get_longest_common_prefix(text) {
                Some((pre, tok)) => (*tok, pre.len()),
                None => (self.bytes[text[0] as usize], 1),
            };
            tokens.push(tok);
            text = &text[len..]
        }

        tokens
    }
    #[inline]
    fn decode(&self, token: utok) -> Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(self.token(token))
    }
}

static MAP_UTF8_TO_BYTE: LazyLock<HashMap<String, u8>> = LazyLock::new(unicode_utf8_to_byte_map);

fn unicode_utf8_to_byte_map() -> HashMap<String, u8> {
    let mut map = HashMap::with_capacity(256);

    for ch in 0x21..=0x7E {
        map.insert(unicode_cpt_to_utf8(ch as _), ch);
    }

    for ch in 0xA1..=0xAC {
        map.insert(unicode_cpt_to_utf8(ch as _), ch);
    }

    for ch in 0xAE..=0xFF {
        map.insert(unicode_cpt_to_utf8(ch as _), ch);
    }

    let mut n = 0u32;
    for ch in 0..256 {
        let piece = unicode_cpt_to_utf8(ch as _);
        if !map.contains_key(&piece) {
            map.insert(unicode_cpt_to_utf8(256 + n), ch as _);
            n += 1;
        }
    }

    map
}

fn unicode_cpt_to_utf8(cpt: u32) -> String {
    let mut bytes = Vec::new();

    if cpt <= 0x7F {
        // 1-byte UTF-8
        bytes.push(cpt as u8);
    } else if cpt <= 0x7FF {
        // 2-byte UTF-8
        bytes.push(((cpt >> 6) & 0x1F) as u8 | 0xC0);
        bytes.push((cpt & 0x3F) as u8 | 0x80);
    } else if cpt <= 0xFFFF {
        // 3-byte UTF-8
        bytes.push(((cpt >> 12) & 0x0F) as u8 | 0xE0);
        bytes.push(((cpt >> 6) & 0x3F) as u8 | 0x80);
        bytes.push((cpt & 0x3F) as u8 | 0x80);
    } else if cpt <= 0x10FFFF {
        // 4-byte UTF-8
        bytes.push(((cpt >> 18) & 0x07) as u8 | 0xF0);
        bytes.push(((cpt >> 12) & 0x3F) as u8 | 0x80);
        bytes.push(((cpt >> 6) & 0x3F) as u8 | 0x80);
        bytes.push((cpt & 0x3F) as u8 | 0x80);
    } else {
        panic!()
    }

    String::from_utf8(bytes).unwrap()
}
