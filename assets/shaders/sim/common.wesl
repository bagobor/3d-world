// Shared by every simulation pass: the globals uniform, cube-sphere addressing and
// hashing/noise utilities.
//
// Cube-sphere convention (must match `src/planet/mesh.rs`):
//   face f has a normal axis N, and in-plane axes U and V. A texel (x, y) on face f maps to
//   s = (x + 0.5) / res * 2 - 1   (likewise t for y)
//   a = tan(s * PI / 4)           (equal-angle warp: texels are within ~1.4x of equal area)
//   dir = normalize(N + a * U + b * V)
// and back with the dominant axis of dir and atan.

struct SimGlobals {
    resolution: u32,
    step: u32,
    seed: u32,
    total_steps: u32,
    dt: f32,
    time: f32,
    inv_resolution: f32,
    _pad: f32,
}

const PI: f32 = 3.14159265358979;
const QUARTER_PI: f32 = 0.785398163397448;

fn face_normal(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3(1.0, 0.0, 0.0); }
        case 1u: { return vec3(-1.0, 0.0, 0.0); }
        case 2u: { return vec3(0.0, 1.0, 0.0); }
        case 3u: { return vec3(0.0, -1.0, 0.0); }
        case 4u: { return vec3(0.0, 0.0, 1.0); }
        default: { return vec3(0.0, 0.0, -1.0); }
    }
}

fn face_u(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3(0.0, 0.0, -1.0); }
        case 1u: { return vec3(0.0, 0.0, 1.0); }
        case 2u, 3u, 4u: { return vec3(1.0, 0.0, 0.0); }
        default: { return vec3(-1.0, 0.0, 0.0); }
    }
}

fn face_v(face: u32) -> vec3<f32> {
    switch face {
        case 2u: { return vec3(0.0, 0.0, -1.0); }
        case 3u: { return vec3(0.0, 0.0, 1.0); }
        default: { return vec3(0.0, 1.0, 0.0); }
    }
}

// Face-local coordinate in [-1, 1]^2 (pre-warp) to unit direction.
fn face_st_to_dir(face: u32, st: vec2<f32>) -> vec3<f32> {
    let ab = tan(st * QUARTER_PI);
    return normalize(face_normal(face) + ab.x * face_u(face) + ab.y * face_v(face));
}

// Continuous texel coordinate (texel centres at integer + 0.5) to unit direction. Works for
// coordinates slightly outside the face, which is how cross-face neighbours are found.
fn texel_to_dir(face: u32, texel: vec2<f32>, inv_res: f32) -> vec3<f32> {
    return face_st_to_dir(face, texel * inv_res * 2.0 - 1.0);
}

fn dir_to_face(dir: vec3<f32>) -> u32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z {
        return select(1u, 0u, dir.x >= 0.0);
    }
    if a.y >= a.z {
        return select(3u, 2u, dir.y >= 0.0);
    }
    return select(5u, 4u, dir.z >= 0.0);
}

// Unit direction on a known face to face-local coordinate in [-1, 1]^2 (pre-warp).
fn dir_to_face_st(face: u32, dir: vec3<f32>) -> vec2<f32> {
    let n = dot(dir, face_normal(face));
    let ab = vec2(dot(dir, face_u(face)), dot(dir, face_v(face))) / n;
    return atan(ab) / QUARTER_PI;
}

// Integer texel holding a direction: (x, y, face).
fn dir_to_texel(dir: vec3<f32>, res: u32) -> vec3<u32> {
    let face = dir_to_face(dir);
    let st = dir_to_face_st(face, dir);
    let xy = clamp(vec2<i32>(floor((st * 0.5 + 0.5) * f32(res))), vec2(0), vec2(i32(res) - 1));
    return vec3(vec2<u32>(xy), face);
}

// Neighbour of a texel by an integer offset, wrapping across cube edges. Returns
// (x, y, face). Valid for small offsets (|offset| << res).
fn neighbor_texel(texel: vec3<u32>, offset: vec2<i32>, res: u32) -> vec3<u32> {
    let p = vec2<i32>(texel.xy) + offset;
    if all(p >= vec2(0)) && all(p < vec2(i32(res))) {
        return vec3(vec2<u32>(p), texel.z);
    }
    let dir = texel_to_dir(texel.z, vec2<f32>(p) + 0.5, 1.0 / f32(res));
    return dir_to_texel(dir, res);
}

// ---------------------------------------------------------------------------------------
// Hashing and noise

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg3(v: vec3<u32>) -> vec3<u32> {
    var w = v * 1664525u + 1013904223u;
    w.x += w.y * w.z;
    w.y += w.z * w.x;
    w.z += w.x * w.y;
    w ^= w >> vec3(16u);
    w.x += w.y * w.z;
    w.y += w.z * w.x;
    w.z += w.x * w.y;
    return w;
}

// Random unit-ish gradient for an integer lattice point.
fn lattice_gradient(cell: vec3<i32>, seed: u32) -> vec3<f32> {
    let h = pcg3(bitcast<vec3<u32>>(cell) ^ vec3(seed, seed * 3u + 1u, seed * 7u + 2u));
    return vec3<f32>(h & vec3(0xffffu)) / 32767.5 - 1.0;
}

// 3D gradient noise, roughly in [-1, 1].
fn gradient_noise(p: vec3<f32>, seed: u32) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var n: array<f32, 8>;
    for (var k = 0u; k < 8u; k++) {
        let o = vec3<i32>(vec3(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u));
        n[k] = dot(lattice_gradient(i + o, seed), f - vec3<f32>(o));
    }
    let x0 = mix(mix(n[0], n[1], u.x), mix(n[2], n[3], u.x), u.y);
    let x1 = mix(mix(n[4], n[5], u.x), mix(n[6], n[7], u.x), u.y);
    return mix(x0, x1, u.z) * 1.6;
}
