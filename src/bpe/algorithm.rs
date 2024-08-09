use super::{utok, Bpe};
use std::{
    cmp::Ordering::{self, Equal},
    collections::BinaryHeap,
    fmt,
    iter::zip,
    ops::Range,
};

pub struct MergeState<'v, 't> {
    text: &'t [u8],
    bpe: &'v Bpe,
    marks: Vec<Mark>,
    merges: BinaryHeap<Merge>,
}

pub struct IntoIter<'v> {
    bpe: &'v Bpe,
    marks: Vec<Mark>,
    i: usize,
}

pub struct Iter<'a> {
    bpe: &'a Bpe,
    marks: &'a [Mark],
}

impl Bpe {
    pub fn begin_merge<'v, 't>(&'v self, text: &'t str) -> MergeState<'v, 't> {
        let mut marks = vec![Mark::unk(self.unk); text.len()];
        let mut merges = BinaryHeap::new();

        let mut buf = [0u8; 4];
        let mut last = None;
        for (i, c) in text.char_indices() {
            let c = c.encode_utf8(&mut buf).as_bytes();
            last = if let Some(token) = self.find_piece(c) {
                marks[i].token = token;
                if let Some(pos) = last.take() {
                    marks[i].back_distance = (i - pos) as _;
                    if let Some(merge) = self.build_merge(
                        text.as_bytes(),
                        pos..i + c.len(),
                        (marks[pos].token, token),
                    ) {
                        merges.push(merge);
                    }
                }
                Some(i)
            } else {
                for (&b, mark) in zip(c, &mut marks[i..]) {
                    mark.token = self.bytes[b as usize];
                }
                None
            };
        }

        MergeState {
            text: text.as_bytes(),
            bpe: self,
            marks,
            merges,
        }
    }

    fn build_merge(&self, text: &[u8], range: Range<usize>, pair: (utok, utok)) -> Option<Merge> {
        self.find_piece(&text[range.clone()]).map(|merged| Merge {
            pos: range.start,
            pair,
            merge: merged,
            rank: self.token(merged).rank,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Mark {
    token: utok,
    back_distance: u32,
}

impl Mark {
    #[inline(always)]
    const fn unk(unk: utok) -> Self {
        Self {
            token: unk,
            back_distance: 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Merge {
    pos: usize,
    pair: (utok, utok),
    merge: utok,
    rank: u32,
}
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        // 比较顺序：rank -> merged -> pos -> pair
        match self.rank.cmp(&other.rank) {
            Equal => match self.merge.cmp(&other.merge) {
                Equal => match self.pos.cmp(&other.pos) {
                    Equal => self.pair.cmp(&other.pair),
                    other => other,
                },
                other => other,
            },
            other => other,
        }
        .reverse()
    }
}
impl PartialOrd for Merge {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl MergeState<'_, '_> {
    /// 尝试执行一次合并，返回是否成功执行了一次合并。
    pub fn merge(&mut self) -> bool {
        // 一次合并将涉及至多 4 个 token：
        //
        // t0 t1 t2 t3
        // -- -- -- --
        //      ↓
        // t0 merge t3
        // -- ----- --
        //
        // 成功的合并将至少消费合并队列中的 1 个项，
        // 同时至多向合并队列添加 2 个项：
        //
        // t0 merge t3
        //    --------
        // --------

        // 从合并队列消费
        while let Some(Merge {
            pos: p1,
            pair: (t1, t2),
            merge,
            ..
        }) = self.merges.pop()
        {
            // 确认合并项有效性
            if self.marks[p1].token != t1 {
                continue;
            }
            let l1 = self.bpe.token(t1).len();
            let p2 = p1 + l1;
            if self.marks[p2].token != t2 {
                continue;
            }
            // 合并
            self.marks[p1].token = merge;
            self.marks[p2].token = self.bpe.unk;

            let l2 = self.bpe.token(t2).len();
            let p3 = p2 + l2;
            // 创建 merge + t3 合并项
            match self.marks.get_mut(p3) {
                None => {}
                Some(Mark {
                    token,
                    back_distance,
                }) => {
                    *back_distance = (l1 + l2) as _;

                    let t3 = *token;
                    let l3 = self.bpe.token(t3).len();
                    let p4 = p3 + l3;
                    if let Some(merge) = self.bpe.build_merge(self.text, p1..p4, (merge, t3)) {
                        self.merges.push(merge);
                    }
                }
            }
            // 创建 t0 + merge 合并项
            match self.marks[p1].back_distance as usize {
                0 => {}
                l0 => {
                    let p0 = p1 - l0;
                    let t0 = self.marks[p0].token;
                    if let Some(merge) = self.bpe.build_merge(self.text, p0..p3, (t0, merge)) {
                        self.merges.push(merge);
                    }
                }
            }
            // 成功合并
            return true;
        }
        false
    }

    #[inline]
    pub fn iter(&self) -> Iter {
        Iter {
            bpe: self.bpe,
            marks: &self.marks,
        }
    }
}

impl<'v> IntoIterator for MergeState<'v, '_> {
    type Item = utok;
    type IntoIter = IntoIter<'v>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            bpe: self.bpe,
            marks: self.marks,
            i: 0,
        }
    }
}

impl Iterator for IntoIter<'_> {
    type Item = utok;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.marks[self.i..] {
            &[Mark { token, .. }, ..] => {
                self.i += self.bpe.token(token).len();
                Some(token)
            }
            [] => None,
        }
    }
}

impl Iterator for Iter<'_> {
    type Item = utok;

    fn next(&mut self) -> Option<Self::Item> {
        match self.marks {
            &[Mark { token, .. }, ref tail @ ..] => {
                self.marks = &tail[self.bpe.token(token).len() - 1..];
                Some(token)
            }
            [] => None,
        }
    }
}

impl fmt::Display for MergeState<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::str::{from_utf8, from_utf8_unchecked};

        writeln!(f, "---------------------------")?;
        {
            writeln!(f, "text:")?;
            writeln!(f, "  {}", unsafe { from_utf8_unchecked(self.text) })?;
        }
        writeln!(f, "---------------------------")?;
        {
            writeln!(f, "tokens:")?;
            write!(f, "  ")?;
            for token in self.iter() {
                let text = unsafe { from_utf8_unchecked(self.bpe.token(token)) };
                write!(f, "{text}")?;
            }
            writeln!(f)?;
        }
        writeln!(f, "---------------------------")?;
        {
            writeln!(f, "tokens:")?;
            for token in self.iter() {
                write!(f, "  {token:>6}: ")?;
                match from_utf8(self.bpe.token(token)) {
                    Ok(s) => writeln!(f, "{s}")?,
                    Err(_) => writeln!(f, "{token:?}")?,
                }
            }
        }
        writeln!(f, "---------------------------")?;
        {
            writeln!(f, "merges:")?;
            let mut merges = self.merges.clone();
            while let Some(Merge {
                rank,
                merge: merged,
                ..
            }) = merges.pop()
            {
                let text = unsafe { from_utf8_unchecked(self.bpe.token(merged)) };
                writeln!(f, "  {rank:>6} | {text}")?;
            }
        }
        writeln!(f, "---------------------------")
    }
}
