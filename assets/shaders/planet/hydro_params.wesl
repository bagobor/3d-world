// Params shared by the hydraulic erosion passes. Mirrors `HydroParams` in `src/planet/mod.rs`.
//
// Units: heights and water depth in metres; one step is one unit of sim time. Water flux is
// the depth of water (over the source cell) that moves to a neighbour per step.

struct HydroParams {
    planet_radius_m: f32,
    rain_m: f32,            // added to every land cell each step
    evaporation: f32,       // fraction of water lost per step
    pipe_gain: f32,         // flux gained per metre of water-surface difference
    flux_damping: f32,      // fraction of last step's flux kept (momentum)
    capacity: f32,          // sediment capacity per (discharge x slope)
    erosion_rate: f32,      // fraction of the capacity deficit picked up per step
    deposition_rate: f32,   // fraction of the surplus dropped per step
    min_slope: f32,         // keeps flat rivers able to carry some sediment
    max_erosion_m: f32,     // per-step clamp on terrain lowered by water
    discharge_smoothing: f32, // EMA factor for the rendered/used discharge
    _pad: f32,
}
