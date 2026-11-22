//! Baked fields on the CPU, and the disk cache for generated planets.
//!
//! When the generate phase finishes, every field marked with [`SimGraph::bake`] is read back
//! from the GPU into [`BakedFields`] (CPU-side consumers such as the LOD terrain read them
//! there) and, if the cache is enabled, written to `<dir>/<tag>-seed<seed>-res<res>.bin`. On start-up the same
//! file, if present and valid, is uploaded straight into the field textures and generation is
//! skipped. `tag` names the generator version: bump it whenever generation changes, so stale
//! planets are not loaded.
//!
//! File layout (little endian): magic `PLNTSIM1`, `u32` resolution, `u32` seed, `u32` field
//! count, then per field: `u32` name length, name bytes, `u32` bytes per texel, `u64` data
//! length, tightly packed texels for the 6 cube faces in order.

use std::{
    fs,
    io::{self, Read, Write},
    path::PathBuf,
};

use bevy::{
    platform::collections::HashMap,
    prelude::*,
    render::gpu_readback::{ReadbackComplete, ReadbackOnce},
};

use super::{CUBE_FACES, SimControl, SimFields, SimGraph, SimStats};

const MAGIC: &[u8; 8] = b"PLNTSIM1";

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub dir: PathBuf,
    /// Generator version; part of the file name.
    pub tag: String,
    /// Load an existing file. `false` always generates (and overwrites the file).
    pub reuse: bool,
}

/// CPU copy of the current planet's baked fields: tightly packed texels for the 6 cube faces
/// in order, one entry per baked field. Filled after generation or a cache load; `version`
/// changes every time the planet does.
#[derive(Resource, Default)]
pub struct BakedFields {
    pub resolution: u32,
    pub seed: u32,
    pub version: u64,
    pub fields: HashMap<&'static str, Vec<u8>>,
}

impl BakedFields {
    pub fn ready(&self) -> bool {
        self.version > 0
    }

    /// A baked R32Float field as floats.
    pub fn f32_field(&self, name: &str) -> Option<Vec<f32>> {
        let bytes = self.fields.get(name)?;
        Some(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        )
    }

    pub(super) fn set(
        &mut self,
        resolution: u32,
        seed: u32,
        fields: HashMap<&'static str, Vec<u8>>,
    ) {
        self.resolution = resolution;
        self.seed = seed;
        self.fields = fields;
        self.version += 1;
    }
}

#[derive(Resource)]
pub struct SimCache {
    config: Option<CacheConfig>,
    /// Generation runs whose readback has been requested.
    requested: u64,
    /// Seed of the generation being saved.
    pending_seed: u32,
    pending: Vec<(&'static str, Vec<u8>)>,
    saved: u64,
    last_saved: Option<PathBuf>,
}

impl SimCache {
    pub(super) fn new(config: Option<CacheConfig>) -> Self {
        Self {
            config,
            requested: 0,
            pending_seed: 0,
            pending: Vec::new(),
            saved: 0,
            last_saved: None,
        }
    }

    pub fn enabled(&self) -> bool {
        self.config.is_some()
    }

    /// Generation runs written to disk so far.
    pub fn saved_generations(&self) -> u64 {
        self.saved
    }

    pub fn last_saved(&self) -> Option<&PathBuf> {
        self.last_saved.as_ref()
    }

    pub fn path(&self, resolution: u32, seed: u32) -> Option<PathBuf> {
        self.config.as_ref().map(|c| {
            c.dir
                .join(format!("{}-seed{seed}-res{resolution}.bin", c.tag))
        })
    }

    /// Baked field data for `seed`, or `None` if there is no valid cache file.
    pub(super) fn load(
        &mut self,
        graph: &SimGraph,
        seed: u32,
    ) -> Option<HashMap<&'static str, Vec<u8>>> {
        let path = self.path(graph.resolution, seed)?;
        if !self.config.as_ref()?.reuse {
            info!("sim cache: regenerating {}", path.display());
            return None;
        }
        if !path.exists() {
            info!("sim cache: no planet at {}, generating", path.display());
            return None;
        }
        match read_file(&path, graph, seed) {
            Ok(fields) => {
                info!("sim cache: loaded planet from {}", path.display());
                Some(fields)
            }
            Err(err) => {
                warn!("sim cache: ignoring {}: {err}", path.display());
                None
            }
        }
    }
}

fn expected_len(graph: &SimGraph, name: &str) -> usize {
    let texel = graph.field(name).unwrap().format.texel_bytes() as usize;
    (graph.resolution as usize).pow(2) * CUBE_FACES as usize * texel
}

fn read_file(
    path: &PathBuf,
    graph: &SimGraph,
    seed: u32,
) -> io::Result<HashMap<&'static str, Vec<u8>>> {
    let bad = |msg: String| io::Error::new(io::ErrorKind::InvalidData, msg);
    let mut r = io::BufReader::new(fs::File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(bad("not a planet cache file".into()));
    }
    let mut u32_ = || -> io::Result<u32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    };
    let (res, file_seed, count) = (u32_()?, u32_()?, u32_()?);
    if res != graph.resolution || file_seed != seed {
        return Err(bad(format!("made for res {res}, seed {file_seed}")));
    }
    let mut out = HashMap::default();
    for _ in 0..count {
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4)?;
        let mut name = vec![0u8; u32::from_le_bytes(b4) as usize];
        r.read_exact(&mut name)?;
        r.read_exact(&mut b4)?; // bytes per texel (validated through the length)
        let mut b8 = [0u8; 8];
        r.read_exact(&mut b8)?;
        let mut data = vec![0u8; u64::from_le_bytes(b8) as usize];
        r.read_exact(&mut data)?;
        let name = String::from_utf8_lossy(&name).into_owned();
        if let Some(&known) = graph.baked.iter().find(|n| **n == name) {
            if data.len() != expected_len(graph, known) {
                return Err(bad(format!("field `{name}` has the wrong size")));
            }
            out.insert(known, data);
        }
    }
    if let Some(missing) = graph.baked.iter().find(|n| !out.contains_key(*n)) {
        return Err(bad(format!("field `{missing}` missing")));
    }
    Ok(out)
}

fn write_file(
    path: &PathBuf,
    resolution: u32,
    seed: u32,
    fields: &[(&str, Vec<u8>)],
    graph: &SimGraph,
) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut w = io::BufWriter::new(fs::File::create(&tmp)?);
        w.write_all(MAGIC)?;
        w.write_all(&resolution.to_le_bytes())?;
        w.write_all(&seed.to_le_bytes())?;
        w.write_all(&(fields.len() as u32).to_le_bytes())?;
        for (name, data) in fields {
            let texel = graph.field(name).unwrap().format.texel_bytes();
            w.write_all(&(name.len() as u32).to_le_bytes())?;
            w.write_all(name.as_bytes())?;
            w.write_all(&texel.to_le_bytes())?;
            w.write_all(&(data.len() as u64).to_le_bytes())?;
            w.write_all(data)?;
        }
        w.flush()?;
    }
    fs::rename(tmp, path)
}

/// GPU readbacks pad every row to 256 bytes; strip that.
fn unpad(data: &[u8], resolution: u32, texel_bytes: u32) -> Option<Vec<u8>> {
    let row = (resolution * texel_bytes) as usize;
    let padded = row.next_multiple_of(256);
    let rows = resolution as usize * CUBE_FACES as usize;
    let mut out = Vec::with_capacity(row * rows);
    for i in 0..rows {
        out.extend_from_slice(data.get(i * padded..i * padded + row)?);
    }
    Some(out)
}

/// After each finished generation, reads back the baked fields into [`BakedFields`] and
/// writes the cache file.
pub(super) fn save_baked_fields(
    mut commands: Commands,
    graph: Res<SimGraph>,
    fields: Option<Res<SimFields>>,
    stats: Res<SimStats>,
    control: Res<SimControl>,
    mut cache: ResMut<SimCache>,
    mut baked: ResMut<BakedFields>,
) {
    let Some(fields) = fields else { return };

    // A generation just finished: request readbacks of every baked field.
    let completed = stats.generations_completed();
    if completed > cache.requested && stats.live() {
        cache.requested = completed;
        cache.pending_seed = control.seed;
        cache.pending.clear();
        let resolution = graph.resolution;
        for &name in &graph.baked {
            let texel = graph.field(name).unwrap().format.texel_bytes();
            commands
                .spawn(ReadbackOnce::texture(fields.front(name)))
                .observe(
                    move |event: On<ReadbackComplete>, mut cache: ResMut<SimCache>| match unpad(
                        &event.data,
                        resolution,
                        texel,
                    ) {
                        Some(data) => cache.pending.push((name, data)),
                        None => error!("sim cache: short readback for `{name}`"),
                    },
                );
        }
    }

    // All readbacks in: publish them and write the file.
    if cache.saved < cache.requested && cache.pending.len() == graph.baked.len() {
        let seed = cache.pending_seed;
        let pending = std::mem::take(&mut cache.pending);
        baked.set(graph.resolution, seed, pending.iter().cloned().collect());
        cache.saved = cache.requested;
        let Some(path) = cache.path(graph.resolution, seed) else {
            return;
        };
        match write_file(&path, graph.resolution, seed, &pending, &graph) {
            Ok(()) => {
                let bytes: usize = pending.iter().map(|(_, d)| d.len()).sum();
                info!(
                    "sim cache: saved planet ({:.1} MiB) to {}",
                    bytes as f64 / (1024.0 * 1024.0),
                    path.display()
                );
                cache.last_saved = Some(path);
            }
            Err(err) => error!("sim cache: could not write {}: {err}", path.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::FieldFormat;

    #[test]
    fn unpad_strips_row_padding() {
        // 4 texels of 4 bytes = 16-byte rows, padded to 256.
        let res = 4;
        let rows = res as usize * CUBE_FACES as usize;
        let mut padded = vec![0xAAu8; rows * 256];
        for r in 0..rows {
            for b in 0..16 {
                padded[r * 256 + b] = (r * 16 + b) as u8;
            }
        }
        let out = unpad(&padded, res, 4).unwrap();
        assert_eq!(out.len(), rows * 16);
        assert!(out.iter().enumerate().all(|(i, &v)| v == i as u8));
    }

    #[test]
    fn round_trip() {
        let mut graph = SimGraph::new(4);
        graph
            .add_field("a", FieldFormat::R32Float)
            .add_field("b", FieldFormat::Rgba32Float)
            .add_field("scratch", FieldFormat::R32Float)
            .bake("a")
            .bake("b");
        let a: Vec<u8> = (0..expected_len(&graph, "a")).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..expected_len(&graph, "b"))
            .map(|i| (i * 7) as u8)
            .collect();
        let dir = std::env::temp_dir().join(format!("planet-cache-test-{}", std::process::id()));
        let mut cache = SimCache::new(Some(CacheConfig {
            dir: dir.clone(),
            tag: "test".into(),
            reuse: true,
        }));
        let path = cache.path(4, 9).unwrap();
        write_file(&path, 4, 9, &[("a", a.clone()), ("b", b.clone())], &graph).unwrap();

        let loaded = cache.load(&graph, 9).unwrap();
        assert_eq!(loaded["a"], a);
        assert_eq!(loaded["b"], b);
        // Another seed or resolution is a miss.
        assert!(cache.load(&graph, 10).is_none());
        let mut other = SimGraph::new(8);
        other.add_field("a", FieldFormat::R32Float).bake("a");
        assert!(read_file(&path, &other, 9).is_err());
        fs::remove_dir_all(dir).ok();
    }
}
