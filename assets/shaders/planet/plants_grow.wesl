// Live, every step: plant growth, competition, colonisation and the damage of this step's
// fires.
//
// Each type grows logistically toward its carrying capacity for the local climate. Trees
// shade shrubs and grass, shrubs shade grass, so forests replace grassland where it is wet
// enough, and fire (which removes grass fastest but trees for longest) keeps savannas open.
// Bare land is colonised from neighbouring cover. Living cover drops litter, which is the
// fuel fires burn; litter decomposes faster when wet.
import super::super::sim::common::{SimGlobals, texel_to_dir, neighbor4};
import super::env_params::precipitation;
import super::plant_params::{PlantParams, plantable, capacity};

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: PlantParams;
@group(0) @binding(2) var plants: texture_2d_array<f32>;
@group(0) @binding(3) var fire: texture_2d_array<f32>;
@group(0) @binding(4) var temperature: texture_2d_array<f32>;
@group(0) @binding(5) var moisture: texture_2d_array<f32>;
@group(0) @binding(6) var biome: texture_2d_array<f32>;
@group(0) @binding(7) var animals: texture_2d_array<f32>;
@group(0) @binding(8) var plants_out: texture_storage_2d_array<rgba32float, write>;

fn load_plants(t: vec3<u32>) -> vec4<f32> {
    return textureLoad(plants, vec2<i32>(t.xy), i32(t.z), 0);
}

@compute @workgroup_size(8, 8, 1)
fn plants_grow(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = globals.resolution;
    if id.x >= res || id.y >= res {
        return;
    }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    if !plantable(textureLoad(biome, p, layer, 0).r) {
        textureStore(plants_out, p, layer, vec4(0.0));
        return;
    }
    let dir = texel_to_dir(id.z, vec2<f32>(id.xy) + 0.5, globals.inv_resolution);
    let precip = precipitation(textureLoad(moisture, p, layer, 0).r, dir, params.subtropic_dryness);
    let k = capacity(textureLoad(temperature, p, layer, 0).r, precip);

    let here = load_plants(id);
    var c = here.rgb;
    var fuel = here.a;

    // Shading: what each layer can reach under the layers above it.
    let k_eff = vec3(
        k.x * max(1.0 - 0.85 * c.z - 0.4 * c.y, 0.0),
        k.y * max(1.0 - 0.75 * c.z, 0.0),
        k.z,
    );
    let rate = vec3(params.growth_grass, params.growth_shrub, params.growth_tree);

    var neighbours = vec3(0.0);
    for (var n = 0u; n < 4u; n++) {
        neighbours += load_plants(neighbor4(id, n, res)).rgb;
    }
    neighbours *= 0.25;

    // Logistic growth (decline where cover exceeds what the site supports), colonisation
    // from neighbours and a trickle of seed from far away.
    let room = max(k_eff - c, vec3(0.0));
    c += rate * (2.0 * c * (k_eff - c) + params.spread * neighbours * room) + params.seed_rain * room;

    // Litter: produced by living cover, decomposed faster when wet.
    fuel += params.litter_rate * (c.x + 0.6 * c.y + 0.4 * c.z);
    fuel -= params.decay_rate * (0.3 + precip) * fuel;

    // What the grazers ate this step (`animals_step` ran first and recorded it).
    let eaten = textureLoad(animals, p, layer, 0).zw;
    c.x -= eaten.x;
    c.y -= eaten.y;

    // This step's fire.
    let f = textureLoad(fire, p, layer, 0).r;
    c *= 1.0 - f * vec3(params.burn_grass, params.burn_shrub, params.burn_tree);
    fuel *= 1.0 - f;

    textureStore(plants_out, p, layer, vec4(clamp(c, vec3(0.0), vec3(1.0)), clamp(fuel, 0.0, 2.0)));
}
