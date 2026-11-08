use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, ensure};
use ash::vk;
use tracy_client::{Client, GpuContext, GpuContextType, GpuSpan, SpanLocation};

use crate::gpu::context::RenderingContext;

const QUERIES_PER_FRAME: u32 = 128;

pub struct GpuZone {
    span: GpuSpan,
    begin: u32,
    end: u32,
}

pub struct GpuProfiler {
    ctx: GpuContext,
    pools: Vec<vk::QueryPool>,
    next_query: Vec<u32>,
    pending: Vec<Vec<GpuZone>>,
    valid_mask: u64,
    calib: Option<ash::ext::calibrated_timestamps::Device>,
    last_sync: Instant,
}

impl GpuProfiler {
    pub fn new(context: &RenderingContext, frames_in_flight: usize) -> Result<Self> {
        let device = &context.device;
        let pdev = context.physical_device.handle;
        let gfx = context.queue_families.graphics;

        let period = unsafe { context.instance.get_physical_device_properties(pdev) }
            .limits
            .timestamp_period;
        let bits = unsafe {
            context
                .instance
                .get_physical_device_queue_family_properties(pdev)
        }[gfx as usize]
            .timestamp_valid_bits;
        ensure!(bits > 0, "graphics queue doesn't support timestamps");
        let valid_mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };

        let pools = (0..frames_in_flight)
            .map(|_| unsafe {
                device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(QUERIES_PER_FRAME),
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // One-off submit to get a GPU timestamp that anchors the Tracy context.
        let gpu_start = unsafe {
            let cmd_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(gfx)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )?;
            let cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_reset_query_pool(cmd, pools[0], 0, 1);
            device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pools[0], 0);
            device.end_command_buffer(cmd)?;

            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            device.queue_submit(
                context.queues[gfx as usize],
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                fence,
            )?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;

            let mut ts = [0u64; 1];
            device.get_query_pool_results(
                pools[0],
                0,
                &mut ts,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
            device.destroy_fence(fence, None);
            device.destroy_command_pool(cmd_pool, None);
            (ts[0] & valid_mask) as i64
        };

        let ctx = Client::running()
            .ok_or_else(|| anyhow!("tracy client not started"))?
            .new_gpu_context(
                Some("Graphics Queue"),
                GpuContextType::Vulkan,
                gpu_start,
                period,
            )
            .map_err(|e| anyhow!("tracy gpu context: {e:?}"))?;

        Ok(Self {
            ctx,
            pools,
            next_query: vec![0; frames_in_flight],
            pending: (0..frames_in_flight).map(|_| Vec::new()).collect(),
            valid_mask,
            calib: context.calibrated_timestamps.clone(),
            last_sync: Instant::now(),
        })
    }

    pub fn collect(&mut self, device: &ash::Device, fi: usize) -> Result<()> {
        let n = self.next_query[fi] as usize;
        if n == 0 {
            return Ok(()); // nothing was submitted from this slot
        }
        let mut results = vec![0u64; n];
        unsafe {
            device.get_query_pool_results(
                self.pools[fi],
                0,
                &mut results,
                vk::QueryResultFlags::TYPE_64,
            )?;
        }
        for z in self.pending[fi].drain(..) {
            z.span
                .upload_timestamp_start((results[z.begin as usize] & self.valid_mask) as i64);
            z.span
                .upload_timestamp_end((results[z.end as usize] & self.valid_mask) as i64);
        }
        self.next_query[fi] = 0;
        Ok(())
    }

    pub fn begin_frame(&mut self, device: &ash::Device, cmd: vk::CommandBuffer, fi: usize) {
        unsafe { device.cmd_reset_query_pool(cmd, self.pools[fi], 0, QUERIES_PER_FRAME) };
        self.next_query[fi] = 0;
        self.pending[fi].clear();
    }

    pub fn sync(&mut self) {
        let Some(calib) = &self.calib else { return };
        if self.last_sync.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last_sync = Instant::now();

        let infos =
            [vk::CalibratedTimestampInfoEXT::default().time_domain(vk::TimeDomainEXT::DEVICE)];
        let ts = unsafe {
            let ts_ = calib.get_calibrated_timestamps(&infos);
            if ts_.is_err() {
                return;
            }
            ts_.unwrap()
        };

        self.ctx.sync_gpu_time((ts.0[0] & self.valid_mask) as i64);
    }

    pub fn begin_zone(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        fi: usize,
        loc: &'static SpanLocation,
    ) -> Option<GpuZone> {
        if self.next_query[fi] + 2 > QUERIES_PER_FRAME {
            return None;
        }
        let span = self.ctx.span(loc).ok()?;
        let begin = self.next_query[fi];
        let end = begin + 1;
        self.next_query[fi] += 2;
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                self.pools[fi],
                begin,
            );
        }
        Some(GpuZone { span, begin, end })
    }

    pub fn end_zone(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        fi: usize,
        zone: Option<GpuZone>,
    ) {
        let Some(mut zone) = zone else { return };
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                self.pools[fi],
                zone.end,
            );
        }
        zone.span.end_zone();
        self.pending[fi].push(zone);
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        for p in self.pools.drain(..) {
            unsafe { device.destroy_query_pool(p, None) };
        }
    }
}
