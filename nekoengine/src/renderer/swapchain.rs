use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::vk::{self};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use winit::window::Window;

use std::result::Result::Ok as stdOk;

use crate::gpu::context::{AllocatedImage, RenderingContext, Surface};

pub struct Swapchain {
    allocator: Allocator,
    pub gui_framebuffer: Option<vk::Framebuffer>,
    pub gui_render_pass: Option<vk::RenderPass>,
    pub main_framebuffer: Option<vk::Framebuffer>,
    pub main_render_pass: Option<vk::RenderPass>,
    pub depth_image: AllocatedImage,
    pub resolve_image: AllocatedImage,
    pub msaa_draw_image: AllocatedImage,
    pub desired_image_count: u32,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub image_views: Vec<vk::ImageView>,
    pub images: Vec<vk::Image>,
    pub handle: vk::SwapchainKHR,
    pub window: Arc<Window>,
    pub context: Arc<RenderingContext>,
    pub is_dirty: bool,
}

impl Swapchain {
    pub fn new(context: Arc<RenderingContext>, window: Arc<Window>) -> Result<Self> {
        let surface = &context.surface;
        let (capabilities, formats, present_modes) = surface.query(context.clone())?;
        let format = vk::Format::B8G8R8A8_SRGB;
        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width,
                height: size.height,
            }
        };
        let desired_image_count = (capabilities.min_image_count + 1).clamp(
            capabilities.min_image_count,
            if capabilities.max_image_count == 0 {
                u32::MAX
            } else {
                capabilities.max_image_count
            },
        );

        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: context.instance.clone(),
            device: context.device.clone(),
            physical_device: context.physical_device.handle,
            debug_settings: Default::default(),
            buffer_device_address: context.use_buffer_device_address,
            allocation_sizes: Default::default(),
        })?;

        let extent_3d = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        };
        let msaa_draw_image =
            Self::create_msaa_draw_image(extent_3d, context.clone(), &mut allocator)?;
        let resolve_image = Self::create_resolve_image(extent_3d, context.clone(), &mut allocator)?;
        let depth_image = Self::create_depth_image(extent_3d, context.clone(), &mut allocator)?;

        let mut main_render_pass = None;
        let mut gui_render_pass = None;
        if !context.use_dynamic_rendering {
            main_render_pass = Some(context.create_msaa_render_pass(
                msaa_draw_image.image_format,
                depth_image.image_format,
                context.msaa_samples,
            )?);
            gui_render_pass = Some(context.create_render_pass(
                Some(resolve_image.image_format),
                None,
                vk::AttachmentLoadOp::LOAD,
            )?);
        }

        let mut main_framebuffer = None;
        let mut gui_framebuffer = None;
        if !context.use_dynamic_rendering {
            main_framebuffer = Some(context.create_msaa_framebuffer(
                msaa_draw_image.image_view,
                resolve_image.image_view,
                depth_image.image_view,
                extent,
                main_render_pass.unwrap(),
            )?);
            gui_framebuffer = Some(context.create_framebuffer(
                Some(resolve_image.image_view),
                None,
                extent,
                gui_render_pass.unwrap(),
            )?);
        }

        Ok(Self {
            allocator,
            gui_framebuffer,
            gui_render_pass,
            main_framebuffer,
            main_render_pass,
            depth_image,
            resolve_image,
            msaa_draw_image,
            desired_image_count,
            format,
            extent,
            image_views: Default::default(),
            images: Default::default(),
            handle: vk::SwapchainKHR::null(),
            window,
            context,
            is_dirty: false,
        })
    }

    pub fn resize(&mut self) -> Result<()> {
        let (caps, formats, present_modes) = self.context.surface.query(self.context.clone())?;

        self.extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            let s = self.window.inner_size();
            vk::Extent2D {
                width: s
                    .width
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: s
                    .height
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };

        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(());
        }

        let mut image_count = self.desired_image_count.max(caps.min_image_count);
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }

        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|f| caps.supported_composite_alpha.contains(*f))
        .context("no supported composite alpha")?;

        let pre_transform = vk::SurfaceTransformFlagsKHR::IDENTITY;

        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST;
        anyhow::ensure!(
            caps.supported_usage_flags.contains(usage),
            "surface does not support usage {usage:?}"
        );

        let preferred = [vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB];
        let surface_format = preferred
            .iter()
            .find_map(|p| {
                formats
                    .iter()
                    .find(|f| f.format == *p && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            })
            .or_else(|| formats.first())
            .context("surface reports no formats")?;

        if surface_format.format != self.format {
            tracing::warn!(
                "format {:?} unsupported, using {:?}",
                self.format,
                surface_format.format
            );
            self.format = surface_format.format;
        }

        unsafe {
            self.context.device.device_wait_idle()?;

            let new_swapchain = self
                .context
                .swapchain_extension
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(self.context.surface.handle)
                        .min_image_count(image_count)
                        .image_format(self.format)
                        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                        .image_extent(self.extent)
                        .image_array_layers(1)
                        .image_usage(usage)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(pre_transform)
                        .composite_alpha(composite_alpha)
                        .present_mode(vk::PresentModeKHR::FIFO)
                        .clipped(true)
                        .old_swapchain(self.handle),
                    None,
                )
                .context("create_swapchain")?;

            if let Some(fb) = self.main_framebuffer.take() {
                self.context.device.destroy_framebuffer(fb, None);
            }
            if let Some(fb) = self.gui_framebuffer.take() {
                self.context.device.destroy_framebuffer(fb, None);
            }

            self.image_views.drain(..).for_each(|image_view| {
                self.context.device.destroy_image_view(image_view, None);
            });
            self.images.clear();
            self.context
                .swapchain_extension
                .destroy_swapchain(self.handle, None);

            self.context
                .destroy_image(&mut self.allocator, &mut self.msaa_draw_image);

            self.context
                .destroy_image(&mut self.allocator, &mut self.resolve_image);

            self.context
                .destroy_image(&mut self.allocator, &mut self.depth_image);

            self.handle = new_swapchain;
            self.images = self
                .context
                .swapchain_extension
                .get_swapchain_images(self.handle)?;
            for image in &self.images {
                self.image_views.push(self.context.create_image_view(
                    *image,
                    self.format,
                    vk::ImageAspectFlags::COLOR,
                )?);
            }

            let extent_3d = vk::Extent3D {
                width: self.extent.width,
                height: self.extent.height,
                depth: 1,
            };
            self.msaa_draw_image =
                Self::create_msaa_draw_image(extent_3d, self.context.clone(), &mut self.allocator)?;
            self.resolve_image =
                Self::create_resolve_image(extent_3d, self.context.clone(), &mut self.allocator)?;
            self.depth_image =
                Self::create_depth_image(extent_3d, self.context.clone(), &mut self.allocator)?;

            if !self.context.use_dynamic_rendering {
                let mfb = self.context.create_msaa_framebuffer(
                    self.msaa_draw_image.image_view,
                    self.resolve_image.image_view,
                    self.depth_image.image_view,
                    self.extent,
                    self.main_render_pass.unwrap(),
                )?;
                self.main_framebuffer = Some(mfb);
                let gui_fb = self.context.create_framebuffer(
                    Some(self.resolve_image.image_view),
                    None,
                    self.extent,
                    self.gui_render_pass.unwrap(),
                )?;
                self.gui_framebuffer = Some(gui_fb);
            }
        }

        Ok(())
    }

    fn create_depth_image(
        extent: vk::Extent3D,
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
    ) -> Result<AllocatedImage> {
        context.create_image(
            allocator,
            extent,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
            context.msaa_samples,
        )
    }

    fn create_resolve_image(
        extent: vk::Extent3D,
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
    ) -> Result<AllocatedImage> {
        context.create_image(
            allocator,
            extent,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageAspectFlags::COLOR,
            vk::SampleCountFlags::TYPE_1,
        )
    }

    fn create_gui_draw_image(
        extent: vk::Extent3D,
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
    ) -> Result<AllocatedImage> {
        context.create_image(
            allocator,
            extent,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageAspectFlags::COLOR,
            vk::SampleCountFlags::TYPE_1,
        )
    }

    fn create_msaa_draw_image(
        extent: vk::Extent3D,
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
    ) -> Result<AllocatedImage> {
        context.create_image(
            allocator,
            extent,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageAspectFlags::COLOR,
            context.msaa_samples,
        )
    }

    pub fn check_extent(&mut self) -> Result<()> {
        let caps = unsafe {
            self.context
                .surface_extension
                .get_physical_device_surface_capabilities(
                    self.context.physical_device.handle,
                    self.context.surface.handle,
                )?
        };
        if caps.current_extent.width != u32::MAX && caps.current_extent != self.extent {
            self.is_dirty = true;
        }
        if caps.current_extent.width == 0 || caps.current_extent.height == 0 {
            bail!("Zero extent!")
        }
        Ok(())
    }

    pub fn acquire_next_image(&mut self, sem: vk::Semaphore) -> Result<Option<u32>> {
        let _s = tracy_client::span!("acquire_next_image");
        match unsafe {
            self.context.swapchain_extension.acquire_next_image(
                self.handle,
                u64::MAX,
                sem,
                vk::Fence::null(),
            )
        } {
            Ok((index, _suboptimal)) => Ok(Some(index)), // ignore suboptimal here
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.is_dirty = true;
                Ok(None) // caller skips the frame and resizes
            }
            Err(e) => Err(e).context("acquire_next_image"),
        }
    }

    pub fn present(
        &mut self,
        image_index: u32,
        render_finished_semaphore: vk::Semaphore,
    ) -> Result<()> {
        unsafe {
            match self.context.swapchain_extension.queue_present(
                self.context.queues[self.context.queue_families.present as usize],
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&[render_finished_semaphore])
                    .swapchains(&[self.handle])
                    .image_indices(&[image_index]),
            ) {
                stdOk(is_suboptimal) => is_suboptimal,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
                Err(error) => return Err(error.into()),
            }
        };

        Ok(())
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            if !self.context.use_dynamic_rendering {
                self.context
                    .device
                    .destroy_render_pass(self.main_render_pass.unwrap(), None);
                self.context
                    .device
                    .destroy_render_pass(self.gui_render_pass.unwrap(), None);
            }
            if let Some(fb) = self.main_framebuffer.take() {
                self.context.device.destroy_framebuffer(fb, None);
            }
            if let Some(fb) = self.gui_framebuffer.take() {
                self.context.device.destroy_framebuffer(fb, None);
            }
            self.image_views.drain(..).for_each(|image_view| {
                self.context.device.destroy_image_view(image_view, None);
            });
            self.images.clear();
            self.context
                .swapchain_extension
                .destroy_swapchain(self.handle, None);

            self.context
                .destroy_image(&mut self.allocator, &mut self.msaa_draw_image);

            self.context
                .destroy_image(&mut self.allocator, &mut self.resolve_image);

            self.context
                .destroy_image(&mut self.allocator, &mut self.depth_image);
        }
    }
}
