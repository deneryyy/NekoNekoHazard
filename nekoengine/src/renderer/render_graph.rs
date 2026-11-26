use ash::vk;
use tracy_client::SpanLocation;

use crate::gpu::profiler::{GpuProfiler, GpuZone};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImageAccess {
    ColorAttachmentWrite,
    DepthAttachmentWrite,
    DepthAttachmentRead,
    FragmentSampled,
    ComputeSampled,
    ComputeStorageWrite,
    ComputeStorageRead,
    TransferSrc,
    TransferDst,
    Present,
}

pub struct ImageAccessInfo {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub layout: vk::ImageLayout,
    pub is_write: bool,
}

impl ImageAccess {
    pub fn info(self) -> ImageAccessInfo {
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
        let (stage, access, layout, is_write) = match self {
            Self::ColorAttachmentWrite => (
                S::COLOR_ATTACHMENT_OUTPUT,
                A::COLOR_ATTACHMENT_READ | A::COLOR_ATTACHMENT_WRITE,
                L::COLOR_ATTACHMENT_OPTIMAL,
                true,
            ),
            Self::DepthAttachmentWrite => (
                S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                L::DEPTH_ATTACHMENT_OPTIMAL,
                true,
            ),
            Self::DepthAttachmentRead => (
                S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                A::DEPTH_STENCIL_ATTACHMENT_READ,
                L::DEPTH_READ_ONLY_OPTIMAL,
                false,
            ),
            Self::FragmentSampled => (
                S::FRAGMENT_SHADER,
                A::SHADER_SAMPLED_READ,
                L::SHADER_READ_ONLY_OPTIMAL,
                false,
            ),
            Self::ComputeSampled => (
                S::COMPUTE_SHADER,
                A::SHADER_SAMPLED_READ,
                L::SHADER_READ_ONLY_OPTIMAL,
                false,
            ),
            Self::ComputeStorageWrite => (
                S::COMPUTE_SHADER,
                A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
                L::GENERAL,
                true,
            ),
            Self::ComputeStorageRead => {
                (S::COMPUTE_SHADER, A::SHADER_STORAGE_READ, L::GENERAL, false)
            }
            Self::TransferSrc => (
                S::TRANSFER,
                A::TRANSFER_READ,
                L::TRANSFER_SRC_OPTIMAL,
                false,
            ),
            Self::TransferDst => (
                S::TRANSFER,
                A::TRANSFER_WRITE,
                L::TRANSFER_DST_OPTIMAL,
                true,
            ),
            Self::Present => (S::NONE, A::NONE, L::PRESENT_SRC_KHR, false),
        };
        ImageAccessInfo {
            stage,
            access,
            layout,
            is_write,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BufferAccess {
    ComputeStorageWrite,
    ComputeStorageRead,
    FragmentStorageRead,
    TransferSrc,
    TransferDst,
}

pub struct BufferAccessInfo {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub is_write: bool,
}

impl BufferAccess {
    pub fn info(self) -> BufferAccessInfo {
        use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
        let (stage, access, is_write) = match self {
            Self::ComputeStorageWrite => (
                S::COMPUTE_SHADER,
                A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
                true,
            ),
            Self::ComputeStorageRead => (S::COMPUTE_SHADER, A::SHADER_STORAGE_READ, false),
            Self::FragmentStorageRead => (S::FRAGMENT_SHADER, A::SHADER_STORAGE_READ, false),
            Self::TransferSrc => (S::TRANSFER, A::TRANSFER_READ, false),
            Self::TransferDst => (S::TRANSFER, A::TRANSFER_WRITE, true),
        };
        BufferAccessInfo {
            stage,
            access,
            is_write,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageId(u32);

struct ImageResource {
    image: vk::Image,
    view: vk::ImageView,
    aspect: vk::ImageAspectFlags,
    // tracked while compiling
    layout: vk::ImageLayout,
    last_stage: vk::PipelineStageFlags2,
    last_access: vk::AccessFlags2,
    last_was_write: bool,
}

struct ImageUse {
    id: ImageId,
    access: ImageAccess,
}

struct Pass<'a> {
    name: &'static str,
    location: &'static SpanLocation,
    images: Vec<ImageUse>,
    exec: Box<dyn FnOnce(&PassContext) + 'a>,
}

pub struct ImportState {
    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

impl ImportState {
    pub fn previous_frame() -> Self {
        Self {
            layout: vk::ImageLayout::UNDEFINED,
            stage: vk::PipelineStageFlags2::ALL_COMMANDS,
            access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        }
    }

    pub fn acquired(wait_stage: vk::PipelineStageFlags2) -> Self {
        Self { layout: vk::ImageLayout::UNDEFINED, stage: wait_stage, access: vk::AccessFlags2::NONE }
    }
}

pub struct RenderGraph<'a> {
    images: Vec<ImageResource>,
    passes: Vec<Pass<'a>>,
}

pub struct PassContext<'a> {
    pub device: &'a ash::Device,
    pub cmd: vk::CommandBuffer,
    // plus a way to resolve ImageId -> view/image (a &[ImageResource] works)
}

impl<'a> RenderGraph<'a> {
    pub fn new() -> Self {
        RenderGraph {
            images: Vec::new(),
            passes: Vec::new(),
        }
    }

    pub fn import_image(
        &mut self,
        image: vk::Image,
        view: vk::ImageView,
        aspect: vk::ImageAspectFlags,
        s: ImportState,
    ) -> ImageId {
        self.images.push(ImageResource {
            image,
            view,
            aspect,
            layout: s.layout,
            last_stage: s.stage,
            last_access: s.access,
            last_was_write: true,
        });
        ImageId(self.images.len() as u32 - 1)
    }

    pub fn add_pass(
        &mut self,
        name: &'static str,
        location: &'static SpanLocation,
        setup: impl FnOnce(&mut PassBuilder),
        exec: impl FnOnce(&PassContext) + 'a,
    ) {
        let mut b = PassBuilder { images: vec![] };
        setup(&mut b);
        self.passes.push(Pass {
            name,
            location,
            images: b.images,
            exec: Box::new(exec),
        });
    }

    pub fn execute(
        mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        profiler: &mut GpuProfiler,
        frame_index: usize,
    ) {
        for pass in std::mem::take(&mut self.passes) {
            let _cpu = tracy_client::Client::running().map(|c| {
                c.span_alloc(Some(pass.name), "RenderGraph::execute", file!(), line!(), 0)
            });
            let mut barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();
            let z = profiler.begin_zone(device, cmd, frame_index, &pass.location);

            for u in &pass.images {
                let res = &mut self.images[u.id.0 as usize];
                let new = u.access.info();

                let layout_change = res.layout != new.layout;
                // RAW, WAR, WAW all need a barrier; read-after-read in same layout doesn't
                let hazard = res.last_was_write || new.is_write;

                if layout_change || hazard {
                    barriers.push(
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(res.last_stage)
                            .src_access_mask(res.last_access)
                            .dst_stage_mask(new.stage)
                            .dst_access_mask(new.access)
                            .old_layout(res.layout)
                            .new_layout(new.layout)
                            .image(res.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: res.aspect,
                                base_mip_level: 0,
                                level_count: vk::REMAINING_MIP_LEVELS,
                                base_array_layer: 0,
                                layer_count: vk::REMAINING_ARRAY_LAYERS,
                            }),
                    );
                    // new "epoch": future barriers wait on this pass
                    res.last_stage = new.stage;
                    res.last_access = if new.is_write {
                        new.access
                    } else {
                        vk::AccessFlags2::NONE
                    };
                } else {
                    // another reader: accumulate stages so a later writer waits on all of them
                    res.last_stage |= new.stage;
                }
                res.layout = new.layout;
                res.last_was_write = new.is_write;
            }

            if !barriers.is_empty() {
                unsafe {
                    device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&barriers),
                    );
                }
            }

            let ctx = PassContext { device, cmd };
            (pass.exec)(&ctx);

            profiler.end_zone(device, cmd, frame_index, z);
        }

        // final transitions for imported resources (e.g. swapchain -> PRESENT) go here
    }
}

#[macro_export]
macro_rules! add_pass {
    ($graph:expr, $name:literal, $setup:expr, $exec:expr) => {
        $graph.add_pass($name, tracy_client::span_location!($name), $setup, $exec)
    };
}

pub struct PassBuilder {
    images: Vec<ImageUse>,
}
impl PassBuilder {
    pub fn use_image(&mut self, id: ImageId, access: ImageAccess) {
        self.images.push(ImageUse { id, access });
    }
}
