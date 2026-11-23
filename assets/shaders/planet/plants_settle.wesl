// Live, every step: `plants_shown`, the plant cover the renderers draw. Plants are drawn
// grown in by their local cover, but the live cover isn't steady: grazers and predators
// cycle over a decade or so and take shrub (and some grass and tree) cover with them. At
// time-lapse speed that would make bushes and trees swell and shrink every second, which no
// plant does. So shrubs and trees are drawn at their long-run cover instead: the live cover
// through two smoothing stages of `settle_rate` each (about 25 years), which smooths the
// cycles away while forests still spread, thin and recover on screen.
//
// Fire is the exception: what burns is taken off the shown cover at once, so a fire still
// clears the land as it passes. The burnt plants then regrow on screen as fast as the live
// cover really regrows, back up to the long-run cover; dips of the live cover below the
// shown one (the cycles) don't take it down again. Grass regrows within months, so it is
// shown as it is.
//
// `plants_shown`: r grass, g shrubs, b trees (as in `plants`), a unused.
// `plants_lag`: the two smoothing stages of shrubs and trees: r, g the first; b, a the
// second, the long-run cover.
import super::super::sim::common::SimGlobals;
import super::plant_params::PlantParams;

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: PlantParams;
@group(0) @binding(2) var plants: texture_2d_array<f32>;
@group(0) @binding(3) var fire: texture_2d_array<f32>;
@group(0) @binding(4) var plants_lag: texture_2d_array<f32>;
@group(0) @binding(5) var plants_shown: texture_2d_array<f32>;
@group(0) @binding(6) var lag_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(7) var shown_out: texture_storage_2d_array<rgba32float, write>;

// Once, at live step 0 (also after a cache load): show the cover as it is.
@compute @workgroup_size(8, 8, 1)
fn plants_settle_init(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= globals.resolution || id.y >= globals.resolution {
        return;
    }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    let now = textureLoad(plants, p, layer, 0);
    textureStore(lag_out, p, layer, vec4(now.gb, now.gb));
    textureStore(shown_out, p, layer, vec4(now.rgb, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn plants_settle(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= globals.resolution || id.y >= globals.resolution {
        return;
    }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    let now = textureLoad(plants, p, layer, 0);
    let lag = textureLoad(plants_lag, p, layer, 0);
    // The long-run cover.
    let first = lag.rg + params.settle_rate * (now.gb - lag.rg);
    let mean = lag.ba + params.settle_rate * (first - lag.ba);
    // This step's fire, which plants_grow has just taken off the live cover.
    let f = textureLoad(fire, p, layer, 0).r;
    let burnt = 1.0 - f * vec2(params.burn_shrub, params.burn_tree);
    var shown = textureLoad(plants_shown, p, layer, 0).gb * burnt;
    // Regrow with the live cover, up to the long-run cover; above it (after the long-run
    // cover has fallen), follow it down slowly.
    shown = max(shown, min(now.gb, mean));
    shown += params.settle_rate * min(mean - shown, vec2(0.0));
    textureStore(lag_out, p, layer, vec4(first, mean));
    textureStore(shown_out, p, layer, vec4(now.r, shown, 0.0));
}
