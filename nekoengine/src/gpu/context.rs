use std::{collections::HashSet, sync::Arc};

use anyhow::{Context, Ok, Result};
use ash::ext::debug_utils;
use ash::vk::{
    self, BufferCreateInfo, CommandBufferUsageFlags, PipelineDepthStencilStateCreateInfo,
    RenderingAttachmentInfo,
};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};
use std::ffi::{CStr, c_void};
use tracing::info;
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

#[derive(Clone)]
pub struct QueueFamily {
    pub index: u32,
    pub properties: vk::QueueFamilyProperties,
}

pub struct QueueFamilies {
    pub graphics: u32,
    pub present: u32,
    pub transfer: u32,
    pub compute: u32,
}

pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub properties2: vk::PhysicalDeviceProperties2<'static>,
    pub vulkan12_properties: vk::PhysicalDeviceVulkan12Properties<'static>,
    pub features: vk::PhysicalDeviceFeatures,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_families: Vec<QueueFamily>,
}

pub struct Surface {
    pub handle: vk::SurfaceKHR,
}

#[derive(Default)]
pub struct AllocatedImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Allocation,
    pub extent: vk::Extent3D,
    pub image_format: vk::Format,
}

pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
}

pub struct RenderingContext {
    pub msaa_samples: (vk::SampleCountFlags, u32),
    pub calibrated_timestamps: Option<ash::ext::calibrated_timestamps::Device>,
    pub queues: Vec<vk::Queue>,
    pub swapchain_extension: ash::khr::swapchain::Device,
    pub device: ash::Device,
    pub queue_family_indices: HashSet<u32>,
    pub queue_families: QueueFamilies,
    pub physical_device: PhysicalDevice,
    pub surface: Surface,
    pub surface_extension: ash::khr::surface::Instance,
    debug_messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub instance: ash::Instance,
    pub entry: ash::Entry,
}

impl RenderingContext {
    fn pick_queue_families(
        physical_devices: Vec<PhysicalDevice>,
    ) -> Result<(PhysicalDevice, QueueFamilies)> {
        let physical_device = physical_devices.into_iter().next().unwrap();
        let queue_family = physical_device
            .queue_families
            .iter()
            .find(|queue_family| {
                queue_family
                    .properties
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS)
                    && queue_family
                        .properties
                        .queue_flags
                        .contains(vk::QueueFlags::COMPUTE)
            })
            .map(|queue_family| queue_family.index)
            .context("No suitable queue family found")?;

        Ok((
            physical_device,
            QueueFamilies {
                graphics: queue_family,
                present: queue_family,
                transfer: queue_family,
                compute: queue_family,
            },
        ))
    }

    fn max_supported_samples(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        wanted: vk::SampleCountFlags,
    ) -> vk::SampleCountFlags {
        let props = unsafe {
            instance.get_physical_device_image_format_properties(
                physical_device,
                format,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                usage,
                vk::ImageCreateFlags::empty(),
            )
        };

        let std::result::Result::Ok(props) = props else {
            return vk::SampleCountFlags::TYPE_1;
        };

        for candidate in [
            vk::SampleCountFlags::TYPE_8,
            vk::SampleCountFlags::TYPE_4,
            vk::SampleCountFlags::TYPE_2,
        ] {
            if candidate.as_raw() <= wanted.as_raw() && props.sample_counts.contains(candidate) {
                return candidate;
            }
        }
        vk::SampleCountFlags::TYPE_1
    }

    const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

    pub fn new(window: Arc<Window>) -> Result<Self> {
        unsafe {
            let raw_display_handle = window.display_handle()?.as_raw();
            let raw_window_handle = window.window_handle()?.as_raw();

            let entry = ash::Entry::load()?;
            let want_validation = cfg!(debug_assertions);
            let layer_available = want_validation
                && entry
                    .enumerate_instance_layer_properties()?
                    .iter()
                    .any(|l| l.layer_name_as_c_str().ok() == Some(Self::VALIDATION_LAYER));
            let debug_utils_available = entry
                .enumerate_instance_extension_properties(None)?
                .iter()
                .any(|e| e.extension_name_as_c_str().ok() == Some(debug_utils::NAME));

            if want_validation && !layer_available {
                tracing::warn!("validation layer requested but not found in the APK");
            }
            let use_validation = layer_available && debug_utils_available;

            let mut extensions: Vec<*const std::os::raw::c_char> =
                ash_window::enumerate_required_extensions(raw_display_handle)?.to_vec();
            let mut layers: Vec<*const std::os::raw::c_char> = Vec::new();
            if use_validation {
                extensions.push(debug_utils::NAME.as_ptr());
                layers.push(Self::VALIDATION_LAYER.as_ptr());
            }

            let mut messenger_ci = Self::messenger_info();
            let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
            let mut instance_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&extensions)
                .enabled_layer_names(&layers);
            if use_validation {
                instance_info = instance_info.push_next(&mut messenger_ci);
            }

            let instance = entry.create_instance(&instance_info, None)?;

            let debug_messenger = if use_validation {
                let loader = debug_utils::Instance::new(&entry, &instance);
                let messenger =
                    loader.create_debug_utils_messenger(&Self::messenger_info(), None)?;
                Some((loader, messenger))
            } else {
                None
            };

            let surface_extension = ash::khr::surface::Instance::new(&entry, &instance);

            let surface_handle = ash_window::create_surface(
                &entry,
                &instance,
                raw_display_handle,
                raw_window_handle,
                None,
            )?;

            let mut physical_devices = instance
                .enumerate_physical_devices()?
                .into_iter()
                .map(|handle| {
                    let properties = instance.get_physical_device_properties(handle);
                    let mut vulkan12_properties = vk::PhysicalDeviceVulkan12Properties::default();
                    let mut properties2 = vk::PhysicalDeviceProperties2 {
                        p_next: &mut vulkan12_properties as *mut _ as *mut std::os::raw::c_void,
                        ..Default::default()
                    };
                    instance.get_physical_device_properties2(handle, &mut properties2);
                    let features = instance
                        .get_physical_device_features(handle)
                        .sampler_anisotropy(true)
                        .robust_buffer_access(false)
                        .shader_storage_image_multisample(true);
                    let memory_properties = instance.get_physical_device_memory_properties(handle);
                    let queue_family_properties =
                        instance.get_physical_device_queue_family_properties(handle);

                    let queue_families = queue_family_properties
                        .into_iter()
                        .enumerate()
                        .map(|(index, properties)| QueueFamily {
                            index: index as u32,
                            properties,
                        })
                        .collect::<Vec<_>>();

                    PhysicalDevice {
                        handle,
                        properties,
                        properties2,
                        vulkan12_properties,
                        features,
                        memory_properties,
                        queue_families,
                    }
                })
                .collect::<Vec<_>>();

            physical_devices.retain(|device| {
                surface_extension
                    .get_physical_device_surface_support(device.handle, 0, surface_handle)
                    .unwrap_or(false)
            });

            let (physical_device, queue_families) = Self::pick_queue_families(physical_devices)?;

            let surface = Surface::new(surface_handle);

            let queue_family_indices: HashSet<u32> = HashSet::from([
                queue_families.graphics,
                queue_families.present,
                queue_families.transfer,
                queue_families.compute,
            ]);

            let queue_create_infos = queue_family_indices
                .iter()
                .copied()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(index)
                        .queue_priorities(&[1.0])
                })
                .collect::<Vec<_>>();

            let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut v13)
                .push_next(&mut v12);
            instance.get_physical_device_features2(physical_device.handle, &mut f2);

            let device_api = physical_device.properties.api_version;
            assert!(
                device_api > vk::API_VERSION_1_3,
                "This engine supports only vulkan 1.3 or upwards!"
            );

            let available =
                instance.enumerate_device_extension_properties(physical_device.handle)?;
            let has_extension = |name: &CStr| {
                available
                    .iter()
                    .any(|e| e.extension_name_as_c_str().map_or(false, |n| n == name))
            };

            let use_calibrated_timestamps = has_extension(ash::ext::calibrated_timestamps::NAME);
            info!("using calibrated timestamps: {}", use_calibrated_timestamps);
            let color_samples = Self::max_supported_samples(
                &instance,
                physical_device.handle,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::SampleCountFlags::TYPE_4,
            );
            let depth_samples = Self::max_supported_samples(
                &instance,
                physical_device.handle,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::SampleCountFlags::TYPE_4,
            );
            let samples = if color_samples.as_raw() < depth_samples.as_raw() {
                color_samples
            } else {
                depth_samples
            };
            let samples_num = 4;
            assert!(
                samples.contains(vk::SampleCountFlags::TYPE_4),
                "This device doesn't support MSAA"
            );

            let mut extension_names = Vec::from([ash::khr::swapchain::NAME.as_ptr()]);
            if use_calibrated_timestamps {
                extension_names.push(ash::ext::calibrated_timestamps::NAME.as_ptr());
            }

            let mut v13_features = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);
            let mut v12_features = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .descriptor_indexing(true)
                .descriptor_binding_storage_image_update_after_bind(true)
                .descriptor_binding_sampled_image_update_after_bind(true)
                .descriptor_binding_storage_buffer_update_after_bind(true)
                .descriptor_binding_variable_descriptor_count(true)
                .descriptor_binding_partially_bound(true)
                .runtime_descriptor_array(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                .shader_storage_buffer_array_non_uniform_indexing(true);
            let mut device_info = vk::DeviceCreateInfo::default()
                .enabled_features(&physical_device.features)
                .queue_create_infos(&queue_create_infos)
                .enabled_extension_names(&extension_names);
            if device_api >= vk::API_VERSION_1_3 {
                device_info = device_info.push_next(&mut v13_features);
            }
            device_info = device_info.push_next(&mut v12_features);
            let device = instance.create_device(physical_device.handle, &device_info, None)?;

            let swapchain_extension = ash::khr::swapchain::Device::new(&instance, &device);
            let calibrated_timestamps = use_calibrated_timestamps
                .then(|| ash::ext::calibrated_timestamps::Device::new(&instance, &device));

            let queues = queue_family_indices
                .iter()
                .map(|index| device.get_device_queue(*index, 0))
                .collect::<Vec<_>>();

            Ok(Self {
                msaa_samples: (samples, samples_num),
                calibrated_timestamps,
                queues,
                swapchain_extension,
                device,
                queue_family_indices,
                queue_families,
                physical_device,
                surface,
                surface_extension,
                debug_messenger,
                instance,
                entry,
            })
        }
    }

    unsafe extern "system" fn debug_callback(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        ty: vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user: *mut c_void,
    ) -> vk::Bool32 {
        let msg = unsafe {
            let d = &*data;
            if d.p_message.is_null() {
                "<null>".into()
            } else {
                CStr::from_ptr(d.p_message).to_string_lossy()
            }
        };
        if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            tracing::error!("[vulkan {ty:?}] {msg}");
        } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
            tracing::warn!("[vulkan {ty:?}] {msg}");
        } else {
            tracing::info!("[vulkan {ty:?}] {msg}");
        }
        vk::FALSE
    }

    fn messenger_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
        vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(Self::debug_callback))
    }

    pub fn create_image(
        &self,
        allocator: &mut Allocator,
        extent: vk::Extent3D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_flags: vk::ImageAspectFlags,
        samples: vk::SampleCountFlags,
    ) -> Result<AllocatedImage> {
        let image_handle = unsafe {
            self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(extent)
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(samples)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?
        };

        let requirements = unsafe { self.device.get_image_memory_requirements(image_handle) };

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "Image",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            self.device
                .bind_image_memory(image_handle, allocation.memory(), allocation.offset())?
        };

        let image_view = self.create_image_view(image_handle, format, aspect_flags)?;

        Ok(AllocatedImage {
            image: image_handle,
            image_view,
            allocation: allocation,
            image_format: format,
            extent,
        })
    }

    pub fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
    ) -> Result<vk::ImageView> {
        let image_view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(aspect_flags)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    ),
                None,
            )
        }?;

        Ok(image_view)
    }

    pub fn transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        base_mip: u32,
        mip_count: u32,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
        src_stage: vk::PipelineStageFlags2,
        dst_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_access: vk::AccessFlags2,
    ) {
        unsafe {
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(aspect_mask)
                .base_mip_level(base_mip)
                .level_count(mip_count)
                .base_array_layer(0)
                .layer_count(vk::REMAINING_ARRAY_LAYERS);

            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)
                .old_layout(old_layout)
                .new_layout(new_layout)
                .image(image)
                .subresource_range(range);

            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[barrier]),
            );
        }
    }

    pub fn copy_image_to_image(
        &self,
        command_buffer: vk::CommandBuffer,
        source: vk::Image,
        destination: vk::Image,
        src_size: vk::Extent2D,
        dst_size: vk::Extent2D,
    ) {
        unsafe {
            self.device.cmd_blit_image(
                command_buffer,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .base_array_layer(0)
                            .mip_level(0),
                    )
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: src_size.width as i32,
                            y: src_size.height as i32,
                            z: 1,
                        },
                    ])
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .base_array_layer(0)
                            .mip_level(0),
                    )
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: dst_size.width as i32,
                            y: dst_size.height as i32,
                            z: 1,
                        },
                    ])],
                vk::Filter::LINEAR,
            );
        }
    }

    pub fn create_graphics_pipeline<'a>(
        &self,
        extent: vk::Extent2D,
        draw_image_format: Option<vk::Format>,
        depth_format: Option<vk::Format>,
        vertex_shader: vk::ShaderModule,
        fragment_shader: vk::ShaderModule,
        pipeline_layout: vk::PipelineLayout,
        pipeline_cache: vk::PipelineCache,
        vertex_input_state: vk::PipelineVertexInputStateCreateInfo<'a>,
        samples: vk::SampleCountFlags,
    ) -> Result<vk::Pipeline> {
        let entry_point = std::ffi::CString::new("main")?;

        unsafe {
            Ok(self
                .device
                .create_graphics_pipelines(
                    pipeline_cache,
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .module(vertex_shader)
                                .name(&entry_point),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .module(fragment_shader)
                                .name(&entry_point),
                        ])
                        .vertex_input_state(&vertex_input_state)
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewports(&[vk::Viewport::default()
                                    .width(extent.width as f32)
                                    .height(extent.height as f32)
                                    .max_depth(1.0)])
                                .scissors(&[vk::Rect2D::default().extent(extent)]),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .polygon_mode(vk::PolygonMode::FILL)
                                .cull_mode(vk::CullModeFlags::BACK)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                .line_width(1.0),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(samples),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default().attachments(&[
                                vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                                    .blend_enable(false),
                            ]),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(pipeline_layout)
                        .depth_stencil_state(
                            &PipelineDepthStencilStateCreateInfo::default()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                                .depth_bounds_test_enable(false)
                                .stencil_test_enable(false)
                                .min_depth_bounds(0.0)
                                .max_depth_bounds(1.0),
                        )
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(
                                    &draw_image_format.map_or(vec![], |format| vec![format]),
                                )
                                .depth_attachment_format(
                                    depth_format.unwrap_or(vk::Format::UNDEFINED),
                                ),
                        )],
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap())
        }
    }

    pub fn begin_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        color_view: Option<vk::ImageView>,
        depth_view: Option<vk::ImageView>,
        clear_color: vk::ClearColorValue,
        render_area: vk::Rect2D,
        resolve_image: Option<&AllocatedImage>,
        load_op: vk::AttachmentLoadOp,
    ) {
        unsafe {
            self.device.cmd_begin_rendering(
                command_buffer,
                &vk::RenderingInfo::default()
                    .layer_count(1)
                    .color_attachments(&color_view.map_or(vec![], |view| {
                        vec![
                            vk::RenderingAttachmentInfo::default()
                                .image_view(view)
                                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                .clear_value(vk::ClearValue { color: clear_color })
                                .load_op(load_op)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .resolve_mode(
                                    resolve_image.map_or(vk::ResolveModeFlags::NONE, |_| {
                                        vk::ResolveModeFlags::AVERAGE
                                    }),
                                )
                                .resolve_image_view(
                                    resolve_image
                                        .map_or(vk::ImageView::null(), |image| image.image_view),
                                )
                                .resolve_image_layout(
                                    resolve_image.map_or(vk::ImageLayout::UNDEFINED, |_| {
                                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                                    }),
                                ),
                        ]
                    }))
                    .depth_attachment(&depth_view.map_or(
                        RenderingAttachmentInfo::default(),
                        |view| {
                            RenderingAttachmentInfo::default()
                                .image_view(view)
                                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                .clear_value(vk::ClearValue {
                                    depth_stencil: vk::ClearDepthStencilValue {
                                        depth: 1.0,
                                        stencil: 0,
                                    },
                                })
                                .load_op(load_op)
                                .store_op(vk::AttachmentStoreOp::STORE)
                        },
                    ))
                    .render_area(render_area),
            );
        }
    }

    pub fn create_msaa_render_pass(
        &self,
        color_format: vk::Format,
        depth_format: vk::Format,
        samples: vk::SampleCountFlags,
        resolve: bool,
    ) -> Result<vk::RenderPass> {
        let color_attachment = vk::AttachmentDescription::default()
            .format(color_format)
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let resolve_attachment = vk::AttachmentDescription::default()
            .format(color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_attachment = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE) // depth not read after pass
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let resolve_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_ref = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref))
            .resolve_attachments(if resolve {
                std::slice::from_ref(&resolve_ref)
            } else {
                &[]
            })
            .depth_stencil_attachment(&depth_ref);

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let attachments: &[vk::AttachmentDescription] = if resolve {
            &[color_attachment, resolve_attachment, depth_attachment]
        } else {
            &[color_attachment, depth_attachment]
        };
        let dependencies = [dependency];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        let render_pass = unsafe { self.device.create_render_pass(&render_pass_info, None)? };

        Ok(render_pass)
    }

    pub fn create_render_pass(
        &self,
        color_format: Option<vk::Format>,
        depth_format: Option<vk::Format>,
        load_op: vk::AttachmentLoadOp,
    ) -> Result<vk::RenderPass> {
        let color_attachment = color_format.map(|format| {
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(load_op)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        });
        let depth_attachment = depth_format.map(|format| {
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(load_op)
                .store_op(vk::AttachmentStoreOp::DONT_CARE) // depth not read after pass
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        });
        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_ref = vk::AttachmentReference::default()
            .attachment(color_format.map_or(0, |_| 1))
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let mut subpass =
            vk::SubpassDescription::default().pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);
        if depth_format.is_some() {
            subpass = subpass.depth_stencil_attachment(&depth_ref);
        }
        if color_format.is_some() {
            subpass = subpass.color_attachments(std::slice::from_ref(&color_ref));
        }

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let attachments = [color_attachment, depth_attachment]
            .into_iter()
            .filter_map(|x| x)
            .collect::<Vec<vk::AttachmentDescription>>();
        let dependencies = [dependency];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        let render_pass = unsafe { self.device.create_render_pass(&render_pass_info, None)? };

        Ok(render_pass)
    }

    pub fn create_msaa_framebuffer(
        &self,
        color_view: vk::ImageView,
        resolve_view: vk::ImageView,
        depth_view: vk::ImageView,
        extent: vk::Extent2D,
        render_pass: vk::RenderPass,
    ) -> Result<vk::Framebuffer> {
        let attachments = [color_view, resolve_view, depth_view];

        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);

        let framebuffer = unsafe { self.device.create_framebuffer(&framebuffer_info, None)? };

        Ok(framebuffer)
    }

    pub fn create_framebuffer(
        &self,
        color_view: Option<vk::ImageView>,
        depth_view: Option<vk::ImageView>,
        extent: vk::Extent2D,
        render_pass: vk::RenderPass,
    ) -> Result<vk::Framebuffer> {
        let attachments = [color_view, depth_view]
            .into_iter()
            .filter_map(|x| x)
            .collect::<Vec<vk::ImageView>>();

        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);

        let framebuffer = unsafe { self.device.create_framebuffer(&framebuffer_info, None)? };

        Ok(framebuffer)
    }

    pub fn create_buffer(
        &self,
        name: &str,
        allocator: &mut Allocator,
        alloc_size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<AllocatedBuffer> {
        unsafe {
            let buffer = self.device.create_buffer(
                &BufferCreateInfo::default().usage(usage).size(alloc_size),
                None,
            )?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let allocation = allocator.allocate(&AllocationCreateDesc {
                name: name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
            Ok(AllocatedBuffer { buffer, allocation })
        }
    }

    pub fn destroy_buffer(&self, allocator: &mut Allocator, buffer: &mut AllocatedBuffer) {
        allocator
            .free(std::mem::take(&mut buffer.allocation))
            .unwrap();
        unsafe { self.device.destroy_buffer(buffer.buffer, None) };
    }

    pub fn destroy_image(&self, allocator: &mut Allocator, image: &mut AllocatedImage) {
        unsafe {
            self.device.destroy_image(image.image, None);
            self.device.destroy_image_view(image.image_view, None);
            allocator
                .free(std::mem::take(&mut image.allocation))
                .unwrap();
        }
    }
}

impl Drop for RenderingContext {
    fn drop(&mut self) {
        unsafe {
            self.surface_extension
                .destroy_surface(self.surface.handle, None);
            self.device.destroy_device(None);
            if let Some((loader, messenger)) = self.debug_messenger.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

impl Surface {
    pub fn new(handle: vk::SurfaceKHR) -> Self {
        Self { handle }
    }
    pub fn query(
        &self,
        context: Arc<RenderingContext>,
    ) -> Result<(
        vk::SurfaceCapabilitiesKHR,
        Vec<vk::SurfaceFormatKHR>,
        Vec<vk::PresentModeKHR>,
    )> {
        unsafe {
            let capabilities = context
                .surface_extension
                .get_physical_device_surface_capabilities(
                    context.physical_device.handle,
                    self.handle,
                )?;
            let formats = context
                .surface_extension
                .get_physical_device_surface_formats(context.physical_device.handle, self.handle)?;
            let present_modes = context
                .surface_extension
                .get_physical_device_surface_present_modes(
                    context.physical_device.handle,
                    self.handle,
                )?;
            Ok((capabilities, formats, present_modes))
        }
    }
}
