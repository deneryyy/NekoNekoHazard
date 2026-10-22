use crate::app::engine::resources::data::{
    TextureData, MaterialData, MeshData, ModelData, ModelVertex,
};

use anyhow::{Context, Result, bail};
use gltf::image::Format;
use tracing::warn;

pub fn load_model_glb(name: &str, bytes: &[u8]) -> Result<ModelData> {
    let (document, buffers, gltf_images) =
        gltf::import_slice(bytes).expect("failed to import glb");

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

    Ok(ModelData {
        name: name.to_string(),
        images,
        materials,
        meshes,
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

            out.push(MeshData {
                vertices,
                indices,
                material: prim.material().index(), 
                transform: world[node.index()],
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
        other => bail!("unsupported image format {other:?}"), // float formats
    };

    let rgba8 = if img.format == Format::R8G8B8A8 {
        img.pixels // the common case: no copy
    } else {
        let stride = channels * bytes_per_channel;
        let mut out = Vec::with_capacity(img.width as usize * img.height as usize * 4);
        for px in img.pixels.chunks_exact(stride) {
            // for 16-bit channels take the high byte (little-endian => last byte of the channel)
            let c = |i: usize| px[i * bytes_per_channel + bytes_per_channel - 1];
            let rgba = match channels {
                1 => [c(0), c(0), c(0), 255],  // grey
                2 => [c(0), c(0), c(0), c(1)], // grey + alpha
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
