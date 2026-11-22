//! Cube-sphere mesh using the same face convention and equal-angle warp as
//! `shaders/sim/common.wesl`.

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use std::f32::consts::FRAC_PI_4;

/// (normal, u, v) axes per cube face; must match `face_normal/face_u/face_v` in WGSL.
pub const FACE_AXES: [(Vec3, Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::NEG_Z, Vec3::Y),
    (Vec3::NEG_X, Vec3::Z, Vec3::Y),
    (Vec3::Y, Vec3::X, Vec3::NEG_Z),
    (Vec3::NEG_Y, Vec3::X, Vec3::Z),
    (Vec3::Z, Vec3::X, Vec3::Y),
    (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
];

/// Face-local `st` in [-1, 1]^2 to a unit direction.
pub fn face_st_to_dir(face: usize, st: Vec2) -> Vec3 {
    let (n, u, v) = FACE_AXES[face];
    let ab = Vec2::new((st.x * FRAC_PI_4).tan(), (st.y * FRAC_PI_4).tan());
    (n + ab.x * u + ab.y * v).normalize()
}

/// Unit sphere built from six `subdivisions × subdivisions` grids. Vertices along shared
/// cube edges are duplicated per face; the vertex shader displaces by direction, so the
/// copies stay welded.
pub fn cube_sphere(subdivisions: u32) -> Mesh {
    let n = subdivisions.max(1);
    let verts_per_face = ((n + 1) * (n + 1)) as usize;
    let mut positions = Vec::with_capacity(verts_per_face * 6);
    let mut uvs = Vec::with_capacity(verts_per_face * 6);
    let mut indices = Vec::with_capacity((n * n * 6 * 6) as usize);

    for (face, &(normal, u, v)) in FACE_AXES.iter().enumerate() {
        let base = positions.len() as u32;
        for y in 0..=n {
            for x in 0..=n {
                let st = Vec2::new(x as f32, y as f32) / n as f32 * 2.0 - 1.0;
                positions.push(face_st_to_dir(face, st).to_array());
                uvs.push([x as f32 / n as f32, y as f32 / n as f32]);
            }
        }
        // Keep counter-clockwise winding seen from outside regardless of face handedness.
        let flip = u.cross(v).dot(normal) < 0.0;
        for y in 0..n {
            for x in 0..n {
                let i0 = base + y * (n + 1) + x;
                let i1 = i0 + 1;
                let i2 = i0 + n + 1;
                let i3 = i2 + 1;
                if flip {
                    indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
                } else {
                    indices.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
                }
            }
        }
    }

    let normals = positions.clone();
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_axes_are_orthonormal() {
        for (n, u, v) in FACE_AXES {
            assert!(n.dot(u).abs() < 1e-6 && n.dot(v).abs() < 1e-6 && u.dot(v).abs() < 1e-6);
        }
    }

    #[test]
    fn face_centres_point_along_normals() {
        for (face, (n, _, _)) in FACE_AXES.iter().enumerate() {
            assert!(face_st_to_dir(face, Vec2::ZERO).distance(*n) < 1e-6);
        }
    }

    #[test]
    fn shared_edges_meet() {
        // +X face's +u edge (st.x = 1) is the -Z face's -u edge... check via directions:
        // every edge vertex of one face must coincide with some edge vertex of another face.
        let n = 8;
        let edge = |face: usize| -> Vec<Vec3> {
            let mut out = Vec::new();
            for i in 0..=n {
                let t = i as f32 / n as f32 * 2.0 - 1.0;
                for st in [
                    Vec2::new(-1.0, t),
                    Vec2::new(1.0, t),
                    Vec2::new(t, -1.0),
                    Vec2::new(t, 1.0),
                ] {
                    out.push(face_st_to_dir(face, st));
                }
            }
            out
        };
        for f in 0..6 {
            for p in edge(f) {
                let matched = (0..6)
                    .filter(|&g| g != f)
                    .any(|g| edge(g).iter().any(|q| q.distance(p) < 1e-5));
                assert!(matched, "face {f} edge vertex {p} has no partner");
            }
        }
    }

    #[test]
    fn mesh_is_closed_and_outward() {
        let mesh = cube_sphere(4);
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(pos)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!()
        };
        let Some(Indices::U32(idx)) = mesh.indices() else {
            panic!()
        };
        for tri in idx.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|k| Vec3::from(pos[tri[k] as usize]));
            let normal = (b - a).cross(c - a);
            assert!(normal.dot(a + b + c) > 0.0, "triangle faces inward");
        }
    }
}
