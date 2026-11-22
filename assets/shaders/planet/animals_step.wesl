// Live, every step: grazers and predators feed, breed, die and migrate.
import super::super::sim::common::{SimGlobals, neighbor4};
import super::animal_params::{AnimalParams, habitable, forage, grazing};

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: AnimalParams;
@group(0) @binding(2) var animals: texture_2d_array<f32>;
@group(0) @binding(3) var plants: texture_2d_array<f32>;
@group(0) @binding(4) var fire: texture_2d_array<f32>;
@group(0) @binding(5) var biome: texture_2d_array<f32>;
@group(0) @binding(6) var animals_out: texture_storage_2d_array<rgba32float, write>;

fn load_animals(t: vec3<u32>) -> vec4<f32> {
    return textureLoad(animals, vec2<i32>(t.xy), i32(t.z), 0);
}

fn load_plants(t: vec3<u32>) -> vec4<f32> {
    return textureLoad(plants, vec2<i32>(t.xy), i32(t.z), 0);
}

fn is_habitable(t: vec3<u32>) -> bool {
    return habitable(textureLoad(biome, vec2<i32>(t.xy), i32(t.z), 0).r);
}

// Net flow into a cell from a neighbour: diffusion plus upwind movement towards the more
// attractive side.
fn exchange(here: f32, there: f32, appeal_here: f32, appeal_there: f32, diffusion: f32, taxis: f32) -> f32 {
    let d = appeal_here - appeal_there;
    return diffusion * (there - here) + taxis * (there * max(d, 0.0) - here * max(-d, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn animals_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = globals.resolution;
    if id.x >= res || id.y >= res {
        return;
    }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    if !is_habitable(id) {
        textureStore(animals_out, p, layer, vec4(0.0));
        return;
    }
    let here = load_animals(id);
    let food = load_plants(id);
    var h = here.x;
    var c = here.y;

    // Migration between habitable neighbours (water and ice are barriers).
    let appeal_h = forage(food) - 0.3 * h;
    var dh = 0.0;
    var dc = 0.0;
    for (var n = 0u; n < 4u; n++) {
        let t = neighbor4(id, n, res);
        if !is_habitable(t) {
            continue;
        }
        let there = load_animals(t);
        let appeal_there = forage(load_plants(t)) - 0.3 * there.x;
        dh += exchange(h, there.x, appeal_h, appeal_there, params.grazer_diffusion, params.grazer_taxis);
        dc += exchange(c, there.y, h, there.x, params.predator_diffusion, params.predator_taxis);
    }

    // Feeding, breeding and death.
    let eaten = grazing(h, food, params);
    let kills = params.hunt_rate * c * h / (h + params.hunt_half);
    let kills_clamped = min(kills, 0.9 * h);
    h += params.grazer_gain * (eaten.x + 0.5 * eaten.y)
        - params.grazer_death * h
        - params.grazer_crowding * h * h
        - kills_clamped
        + params.immigration * smoothstep(0.1, 0.4, forage(food));
    c += params.predator_gain * kills_clamped - params.predator_death * c;
    h += dh;
    c += dc;

    // Wildfire kills what can't outrun it.
    let f = textureLoad(fire, p, layer, 0).r;
    h *= 1.0 - params.fire_kill * f;
    c *= 1.0 - params.fire_kill * f;

    textureStore(
        animals_out,
        p,
        layer,
        vec4(clamp(h, 0.0, 4.0), clamp(c, 0.0, 4.0), eaten),
    );
}
