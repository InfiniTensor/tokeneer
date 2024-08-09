//! 这个模块提供对词表的预处理功能，这些功能适用于多种不同算法的分词器。

use crate::utok;
use std::{iter::zip, pin::Pin, slice::from_ref};

/// 收集和预处理词表。
///
/// 几乎所有分词器的词表中都包含一般词（*Normal token*）和单字节词（*Byte token*）。
/// 二者在相同的词空间中，但语义不同。
///
/// 一般词表示具有人类语言语义的字符片段，因此也必定具有完整的 utf-8 编码，
/// 例如单个 ASCII 码 `A`、单词 `hello`、词缀 `tion` 或汉语`优`。
/// 单字节词则表示不在词表中的任何内容，被分解为无法由人类理解的字节编码，模型可以尝试解读或简单引用其内容。
/// 因此，单个 ASCII 码作为含语义的一般词或者作为单字节词可能有不同的 2 个词序号，必须分离到不同的空间中索引。
pub(crate) struct CollectedVocab<'s> {
    /// 词序列表，按词序分割存储每个词的字节序列，并对字节词转义
    pub vocabs: Vec<&'s [u8]>,
    /// 词序表中片段的总字节数
    pub total_len: usize,
    /// 字节词到词序号的映射
    pub bytes: Box<[utok; 256]>,
}

impl<'s> CollectedVocab<'s> {
    /// 收集词表，并对字节词进行转义。
    pub fn collect(vocabs: impl IntoIterator<Item = &'s [u8]>, unk: utok) -> Self {
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
        Self {
            vocabs,
            total_len,
            bytes,
        }
    }

    /// 收集词表，根据提示决定一个词是否是单字节词。
    pub fn collect_with_hint(
        vocabs: impl IntoIterator<Item = &'s [u8]>,
        is_byte: impl IntoIterator<Item = bool>,
        unk: utok,
    ) -> Self {
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
        Self {
            vocabs,
            total_len,
            bytes,
        }
    }
}

/// 利用词表中的重复部分压缩词表。
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
