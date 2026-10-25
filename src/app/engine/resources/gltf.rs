use crate::app::engine::resources::data::{
    Channel, Clip, Interp, MaterialData, MeshData, ModelData, ModelVertex, PathKind, Skeleton,
    SkinData, SkinVertex, TextureData, Trs,
};

use anyhow::{Context, Result, bail};
use glam::{Mat4, Quat, Vec3, Vec4};
use gltf::image::Format;
use tracing::warn;

pub fn load_model_glb(name: &str, bytes: &[u8]) -> Result<ModelData> {
    let (document, buffers, gltf_images) = gltf::import_slice(bytes).expect("failed to import glb");

    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .expect("glb contains no scene");
    let mut world = vec![glam::Mat4::IDENTITY; document.nodes().len()];
    for node in scene.nodes() {
        compute_world(&node, glam::Mat4::IDENTITY, &mut world);
    }

    let mut images: Vec<TextureData> = gltf_images
        .into_iter()
        .enumerate()
        .map(|(i, img)| convert_image(img).with_context(|| format!("image {i}")))
        .collect::<Result<_>>()?;

    let mut materials = Vec::new();
    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        materials.push(MaterialData {
            name: material.name().unwrap_or("Unnamed Material").to_string(),
            base_color_image: pbr
                .base_color_texture()
                .map(|t| t.texture().source().index()),
            base_color_factor: pbr.base_color_factor(),
        });
    }

    // Colour space depends on how an image is used, so decide it here, not in the decoder.
    // Base colour (and later emissive) is sRGB. Normal / metal-rough / AO stay linear (the default).
    for m in &materials {
        if let Some(i) = m.base_color_image {
            images[i].srgb = true;
        }
    }

    let mut meshes = Vec::new();
    for node in scene.nodes() {
        collect_meshes(&node, &world, &buffers, &mut meshes);
    }
    let skeleton = load_skeleton(&document);
    let skins = load_skins(&document, &buffers);
    let clips = load_clips(&document, &buffers);

    Ok(ModelData {
        name: name.to_string(),
        images,
        materials,
        meshes,
        skeleton,
        skins,
        clips,
    })
}

fn collect_meshes(
    node: &gltf::Node,
    world: &[glam::Mat4],
    buffers: &[gltf::buffer::Data],
    out: &mut Vec<MeshData>,
) {
    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                warn!("skipping non-triangle primitive ({:?})", prim.mode());
                continue;
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()]));
            let Some(positions) = reader.read_positions() else {
                continue;
            };

            let mut vertices: Vec<ModelVertex> = positions
                .map(|p| ModelVertex {
                    position: p,
                    uv1: 0.0,
                    normal: [0.0; 3],
                    uv2: 0.0,
                })
                .collect();
            if let Some(normals) = reader.read_normals() {
                for (v, n) in vertices.iter_mut().zip(normals) {
                    v.normal = n;
                }
            }
            if let Some(uvs) = reader.read_tex_coords(0) {
                for (v, uv) in vertices.iter_mut().zip(uvs.into_f32()) {
                    v.uv1 = uv[0];
                    v.uv2 = uv[1];
                }
            }
            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..vertices.len() as u32).collect(), // non-indexed primitive
            };

            let skin_stream = match (node.skin(), reader.read_joints(0), reader.read_weights(0)) {
                (Some(skin), Some(j), Some(w)) => {
                    let sv: Vec<SkinVertex> = j
                        .into_u16()
                        .zip(w.into_f32())
                        .map(|(j, w)| {
                            let sum = (w[0] + w[1] + w[2] + w[3]).max(f32::EPSILON);
                            SkinVertex {
                                joints: j.map(|x| x as u32),
                                weights: w.map(|x| x / sum),
                            }
                        })
                        .collect();
                    Some((skin.index(), sv))
                }
                _ => None,
            };

            let transform = if node.skin().is_some() {
                Mat4::IDENTITY
            } else {
                world[node.index()]
            };

            out.push(MeshData {
                vertices,
                indices,
                skin: skin_stream,
                material: prim.material().index(),
                transform,
                node: node.index(),
            });
        }
    }
    for child in node.children() {
        collect_meshes(&child, world, buffers, out);
    }
}

fn compute_world(node: &gltf::Node, parent: glam::Mat4, world: &mut Vec<glam::Mat4>) {
    let w = parent * glam::Mat4::from_cols_array_2d(&node.transform().matrix());
    world[node.index()] = w;
    for child in node.children() {
        compute_world(&child, w, world);
    }
}

fn convert_image(img: gltf::image::Data) -> Result<TextureData> {
    let (channels, bytes_per_channel) = match img.format {
        Format::R8 => (1, 1),
        Format::R8G8 => (2, 1),
        Format::R8G8B8 => (3, 1),
        Format::R8G8B8A8 => (4, 1),
        Format::R16 => (1, 2),
        Format::R16G16 => (2, 2),
        Format::R16G16B16 => (3, 2),
        Format::R16G16B16A16 => (4, 2),
        other => bail!("unsupported image format {other:?}"),
    };

    let rgba8 = if img.format == Format::R8G8B8A8 {
        img.pixels
    } else {
        let stride = channels * bytes_per_channel;
        let mut out = Vec::with_capacity(img.width as usize * img.height as usize * 4);
        for px in img.pixels.chunks_exact(stride) {
            let c = |i: usize| px[i * bytes_per_channel + bytes_per_channel - 1];
            let rgba = match channels {
                1 => [c(0), c(0), c(0), 255],
                2 => [c(0), c(0), c(0), c(1)],
                3 => [c(0), c(1), c(2), 255],
                _ => [c(0), c(1), c(2), c(3)],
            };
            out.extend_from_slice(&rgba);
        }
        out
    };

    Ok(TextureData {
        width: img.width,
        height: img.height,
        rgba: rgba8,
        srgb: false,
    })
}

pub fn load_skeleton(doc: &gltf::Document) -> Skeleton {
    let n = doc.nodes().len();
    let mut parent = vec![None; n];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut rest = Vec::with_capacity(n);

    for node in doc.nodes() {
        let (t, r, s) = node.transform().decomposed();
        rest.push(Trs {
            t: Vec3::from(t),
            r: Quat::from_array(r).normalize(),
            s: Vec3::from(s),
        });
        for c in node.children() {
            parent[c.index()] = Some(node.index());
            children[node.index()].push(c.index());
        }
    }

    // Depth-first from the roots: a node is always pushed before its children.
    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<usize> = (0..n).rev().filter(|&i| parent[i].is_none()).collect();
    while let Some(i) = stack.pop() {
        order.push(i);
        stack.extend(children[i].iter().rev());
    }

    Skeleton {
        parent,
        rest,
        order,
    }
}

pub fn load_skins(doc: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Vec<SkinData> {
    doc.skins()
        .map(|skin| {
            let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
            let reader = skin.reader(|b| Some(&buffers[b.index()]));
            let mut inverse_bind: Vec<Mat4> = reader
                .read_inverse_bind_matrices()
                .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
                .unwrap_or_default();
            // The spec allows omitting them (= identity).
            inverse_bind.resize(joints.len(), Mat4::IDENTITY);
            SkinData {
                joints,
                inverse_bind,
            }
        })
        .collect()
}

pub fn load_clips(doc: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Vec<Clip> {
    use gltf::animation::util::ReadOutputs;

    doc.animations()
        .enumerate()
        .map(|(idx, anim)| {
            let mut duration = 0.0f32;
            let mut channels = Vec::new();

            for ch in anim.channels() {
                let reader = ch.reader(|b| Some(&buffers[b.index()]));
                let (Some(inputs), Some(outputs)) = (reader.read_inputs(), reader.read_outputs())
                else {
                    continue;
                };
                let times: Vec<f32> = inputs.collect();
                if times.is_empty() {
                    continue;
                }
                duration = duration.max(*times.last().unwrap());

                let (path, values): (PathKind, Vec<Vec4>) = match outputs {
                    ReadOutputs::Translations(it) => (
                        PathKind::Translation,
                        it.map(|v| Vec3::from(v).extend(0.0)).collect(),
                    ),
                    ReadOutputs::Scales(it) => (
                        PathKind::Scale,
                        it.map(|v| Vec3::from(v).extend(0.0)).collect(),
                    ),
                    ReadOutputs::Rotations(it) => {
                        (PathKind::Rotation, it.into_f32().map(Vec4::from).collect())
                    }
                    _ => continue, // morph target weights: not supported
                };

                let interp = match ch.sampler().interpolation() {
                    gltf::animation::Interpolation::Linear => Interp::Linear,
                    gltf::animation::Interpolation::Step => Interp::Step,
                    gltf::animation::Interpolation::CubicSpline => Interp::CubicSpline,
                };

                // Reject malformed channels instead of panicking while sampling.
                let expected = if interp == Interp::CubicSpline {
                    times.len() * 3
                } else {
                    times.len()
                };
                if values.len() != expected {
                    tracing::warn!(
                        "animation channel has {} values, expected {expected}",
                        values.len()
                    );
                    continue;
                }

                channels.push(Channel {
                    node: ch.target().node().index(),
                    path,
                    interp,
                    times,
                    values,
                });
            }

            Clip {
                name: anim
                    .name()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("clip{idx}")),
                duration,
                channels,
            }
        })
        .collect()
}
