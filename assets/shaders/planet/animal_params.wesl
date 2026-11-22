// Params for the live animal populations. Mirrors `AnimalParams` in `src/planet/mod.rs`.
//
// The `animals` field holds, per texel: r = grazers, g = predators (population density,
// about 1 = as many grazers as the best pasture feeds), b = grass and a = shrub cover
// eaten this step. One live step is about a month.
//
// Populations follow Lotka-Volterra dynamics with saturating (Holling type II) feeding,
// and migrate: they diffuse, and move up the gradient of their food (grazers towards
// plant cover, predators towards grazers). `animals_step` runs before `plants_grow`, which
// takes what was eaten (b, a) off the plant cover.

struct AnimalParams {
    graze_rate: f32,         // plant cover one unit of grazers eats per step when food is plentiful
    graze_half: f32,         // food at which grazers eat half that
    grazer_gain: f32,        // grazers gained per unit of cover eaten
    grazer_death: f32,       // grazer losses per step (age, disease)
    grazer_crowding: f32,    // extra losses per step at density 1, growing with density
    hunt_rate: f32,          // grazers one unit of predators kills per step when prey is plentiful
    hunt_half: f32,          // grazer density at which predators kill half that
    predator_gain: f32,      // predators gained per grazer killed
    predator_death: f32,     // predator losses per step
    grazer_diffusion: f32,   // random wandering, fraction exchanged with each neighbour per step
    grazer_taxis: f32,       // movement towards better food
    predator_diffusion: f32,
    predator_taxis: f32,     // movement towards prey
    immigration: f32,        // trickle of grazers into empty pasture, keeps it from staying empty
    fire_kill: f32,          // fraction of animals a full-intensity fire kills
    _pad: f32,
}

const BIOME_ICE: f32 = 2.0;

// Land animals live on: not ocean, lake or ice sheet.
fn habitable(biome: f32) -> bool {
    return biome > BIOME_ICE + 0.5;
}

// What grazers eat, by plant type: grass mostly, shrubs some, trees not (too tall).
fn forage(plants: vec4<f32>) -> f32 {
    return plants.x + 0.5 * plants.y;
}

// Grass and shrub cover eaten this step by a grazer density.
fn grazing(grazers: f32, plants: vec4<f32>, p: AnimalParams) -> vec2<f32> {
    let food = forage(plants);
    if food <= 1e-5 {
        return vec2(0.0);
    }
    let eaten = min(p.graze_rate * grazers * food / (food + p.graze_half), 0.9 * food);
    // Split between grass and shrubs in proportion to what is on offer.
    let share = plants.x / food;
    return vec2(eaten * share, eaten * (1.0 - share) * 2.0);
}
