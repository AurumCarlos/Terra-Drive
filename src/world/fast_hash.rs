//! Table de hachage rapide pour les petites cles (identifiants de noeuds, de cellules).
//!
//! Le hachage par defaut de Rust (SipHash) resiste aux attaques mais coute cher ; les cles d'ici sont
//! des entiers du programme lui-meme. On reprend le hachage de rustc (FxHash) : une rotation et une
//! multiplication par mot.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
pub type FastSet<K> = HashSet<K, BuildHasherDefault<FxHasher>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::NodeId;

    #[test]
    fn node_ids_spread_over_the_table() {
        let mut map: FastMap<NodeId, u32> = FastMap::default();
        for x in 0..64 {
            for z in 0..64 {
                map.insert(NodeId { face: 2, lod: 6, x, z }, x * 64 + z);
            }
        }
        assert_eq!(map.len(), 4096);
        assert_eq!(map[&NodeId { face: 2, lod: 6, x: 13, z: 40 }], 13 * 64 + 40);
        // Les hachages des voisins ne se ressemblent pas : peu de collisions sur les bits bas.
        let mut low = std::collections::HashSet::new();
        for x in 0..64u32 {
            let mut h = FxHasher::default();
            std::hash::Hash::hash(&NodeId { face: 0, lod: 9, x, z: 7 }, &mut h);
            low.insert(h.finish() >> 54);
        }
        assert!(low.len() > 50, "{} valeurs distinctes", low.len());
    }
}
