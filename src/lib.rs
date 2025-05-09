#![deny(warnings)]
mod bpe;
mod common;
mod lpe;
mod tokeneer;
mod utils;
mod vocab;
use std::borrow::Cow;

pub use bpe::Bpe;
pub use bpe::Model;
pub use lpe::Lpe;
pub use tokeneer::Tokeneer;
pub use vocab::TokenType;

/// `utok` for token id.
#[allow(non_camel_case_types)]
pub type utok = u32;
//  添加判断终止符的函数
pub trait Method {
    fn unk_token(&self) -> utok;
    fn vocab_size(&self) -> usize;
    fn internal_special(&self) -> impl IntoIterator<Item = (&str, utok)>;
    fn encode(&self, text: &str) -> impl IntoIterator<Item = utok> + '_;
    fn decode(&self, token: utok) -> Cow<[u8]>;
    fn pre_encode<'s>(&self, text: &'s str) -> Cow<'s, str> {
        text.into()
    }
    fn pre_decode<'s>(&self, text: &'s str) -> Cow<'s, str> {
        text.into()
    }
}
