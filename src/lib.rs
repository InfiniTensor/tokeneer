#![deny(warnings)]

mod bpe;
mod functions;
mod lpe;
mod tokeneer;

pub use bpe::Bpe;
pub use lpe::Lpe;
pub use tokeneer::Tokeneer;

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
