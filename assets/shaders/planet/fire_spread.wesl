// Live, every step: wildfire. A texel burns for one step. Lightning starts fires at random,
// and each burning neighbour may ignite a texel; both scale with flammability, which needs
// fuel (litter and grass), dry climate and warmth. A texel that just burned has no fuel
// left, so fires leave firebreaks behind them and burn out.
import super::super::sim::common::{SimGlobals, texel_to_dir, neighbor4};
import super::env_params::precipitation;
import super::plant_params::{PlantParams, plantable, random};

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: PlantParams;
@group(0) @binding(2) var plants: texture_2d_array<f32>;
@group(0) @binding(3) var fire: texture_2d_array<f32>;
@group(0) @binding(4) var temperature: texture_2d_array<f32>;
@group(0) @binding(5) var moisture: texture_2d_array<f32>;
@group(0) @binding(6) var biome: texture_2d_array<f32>;
@group(0) @binding(7) var fire_out: texture_storage_2d_array<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn fire_spread(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = globals.resolution;
    if id.x >= res || id.y >= res {
        return;
    }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    var out = 0.0;
    // Burning now (its fuel is already gone), or nothing to burn.
    if plantable(textureLoad(biome, p, layer, 0).r) && textureLoad(fire, p, layer, 0).r < 0.05 {
        let dir = texel_to_dir(id.z, vec2<f32>(id.xy) + 0.5, globals.inv_resolution);
        let precip = precipitation(textureLoad(moisture, p, layer, 0).r, dir, params.subtropic_dryness);
        let temp = textureLoad(temperature, p, layer, 0).r;
        let veg = textureLoad(plants, p, layer, 0);
        let fuel = veg.a + 0.1 * veg.r;
        let dryness = clamp(1.2 - 1.4 * precip, 0.0, 1.0);
        let flammable = clamp(fuel, 0.0, 1.0) * dryness * smoothstep(-2.0, 18.0, temp);

        var ignite = random(id, globals.step, globals.seed, 0u) < params.ignition_rate * flammable;
        for (var n = 0u; n < 4u; n++) {
            let t = neighbor4(id, n, res);
            let burning = textureLoad(fire, vec2<i32>(t.xy), i32(t.z), 0).r;
            ignite = ignite || (burning > 0.05 && random(id, globals.step, globals.seed, 1u + n) < params.spread_chance * flammable);
        }
        if ignite {
            out = max(flammable, 0.1);
        }
    }
    textureStore(fire_out, p, layer, vec4(out, 0.0, 0.0, 0.0));
}
