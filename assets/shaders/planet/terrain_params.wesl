// Params shared by the terrain passes. Mirrors `TerrainParams` in `src/planet/mod.rs`.

struct TerrainParams {
    planet_radius_m: f32,
    continent_frequency: f32,
    mountain_height_m: f32,
    ocean_depth_m: f32,
    // Thermal erosion
    talus_slope: f32,
    erosion_rate: f32,
    _pad0: f32,
    _pad1: f32,
}
