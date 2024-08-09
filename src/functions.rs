use crate::utok;
use std::{iter::zip, pin::Pin, slice::from_ref};

// 收集词表字符内容和字节 token，同时计算内容总长度
pub(crate) fn collect_vocabs<'s>(
    vocabs: impl IntoIterator<Item = &'s [u8]>,
    unk: utok,
) -> (Vec<&'s [u8]>, Box<[utok; 256]>, usize) {
    let mut bytes = Box::new([unk; 256]);
    let mut total_len = 0;
    let vocabs = vocabs
        .into_iter()
        .enumerate()
        .map(|(i, piece)| {
            let piece = match as_byte_token(piece) {
                Some(b) => {
                    let b = b as usize;
                    bytes[b] = i as _;
                    from_ref(&BYTES[b])
                }
                None => piece,
            };
            total_len += piece.len();
            piece
        })
        .collect();
    (vocabs, bytes, total_len)
}

// 收集词表字符内容和字节 token，同时计算内容总长度
pub(crate) fn collect_vocabs_with_hint<'s>(
    vocabs: impl IntoIterator<Item = &'s [u8]>,
    is_byte: impl IntoIterator<Item = bool>,
    unk: utok,
) -> (Vec<&'s [u8]>, Box<[utok; 256]>, usize) {
    let mut bytes = Box::new([unk; 256]);
    let mut total_len = 0;
    let vocabs = zip(vocabs, is_byte)
        .enumerate()
        .map(|(i, (piece, is_byte))| {
            let piece = if is_byte {
                let b = as_byte_token(piece)
                    .unwrap_or_else(|| panic!("{piece:?} is not a valid byte token"))
                    as usize;
                bytes[b] = i as _;
                from_ref(&BYTES[b])
            } else {
                piece
            };
            total_len += piece.len();
            piece
        })
        .collect();
    (vocabs, bytes, total_len)
}

pub(crate) struct CompressedVocab {
    pub vocabs: Pin<Box<[u8]>>,
    pub slices: Vec<(usize, usize)>,
}

impl CompressedVocab {
    pub fn new(vocabs: &[&[u8]], total_len: usize) -> Self {
        // 创建字符内容缓存
        let mut slices = vec![(0usize, 0usize); vocabs.len()];
        let mut text_buf = Vec::<u8>::with_capacity(total_len);
        let mut indices = (0..vocabs.len()).collect::<Vec<_>>();
        // 对词按内容长度从长到短排序，因为短的内容有可能是长内容的子串，可以避免重复存储相同内容
        indices.sort_unstable_by_key(|&i| -(vocabs[i].len() as isize));
        for i in indices {
            let v = vocabs[i];
            // 查找子串，若存在则复用，否则将新的内容追加到缓存
            let off = memchr::memmem::find(&text_buf, v).unwrap_or_else(|| {
                let off = text_buf.len();
                text_buf.extend(v);
                off
            });
            slices[i] = (off, v.len());
        }
        Self {
            // 锁定字符串内容的位置，以实现安全的自引用
            vocabs: unsafe { Pin::new_unchecked(text_buf.into_boxed_slice()) },
            slices,
        }
    }
}

const BYTES: [u8; 256] = {
    let mut bytes = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        bytes[i] = i as _;
        i += 1;
    }
    bytes
};

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
