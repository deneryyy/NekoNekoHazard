use anyhow::Result;
use ash::vk;

#[derive(Default)]
pub struct DescriptorLayoutBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    flags: Vec<vk::DescriptorBindingFlags>,
}

impl DescriptorLayoutBuilder {
    pub fn new() -> Self { Self::default() }

    pub fn binding(
        mut self,
        binding: u32,
        ty: vk::DescriptorType,
        count: u32,
        stages: vk::ShaderStageFlags,
        flags: vk::DescriptorBindingFlags,
    ) -> Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(ty)
                .descriptor_count(count)
                .stage_flags(stages),
        );
        self.flags.push(flags);
        self
    }

    pub fn key(&self) -> Vec<(u32, i32, u32, u32, u32)> {
        self.bindings.iter().zip(&self.flags).map(|(b, f)| {
            (b.binding, b.descriptor_type.as_raw(), b.descriptor_count,
             b.stage_flags.as_raw(), f.as_raw())
        }).collect()
    }

    pub fn has_update_after_bind(&self) -> bool {
        self.flags.iter().any(|f| f.contains(vk::DescriptorBindingFlags::UPDATE_AFTER_BIND))
    }

    pub fn build(&self, device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
        let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
            .binding_flags(&self.flags);

        let mut info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&self.bindings)
            .push_next(&mut flags_info);

        if self.has_update_after_bind() {
            info = info.flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL);
        }
        Ok(unsafe { device.create_descriptor_set_layout(&info, None)? })
    }
}

#[derive(Default)]
pub struct LayoutCache {
    map: std::collections::HashMap<Vec<(u32, i32, u32, u32, u32)>, vk::DescriptorSetLayout>,
}

impl LayoutCache {
    pub fn get(&mut self, device: &ash::Device, b: DescriptorLayoutBuilder)
        -> Result<vk::DescriptorSetLayout>
    {
        let key = b.key();
        if let Some(l) = self.map.get(&key) { return Ok(*l); }
        let l = b.build(device)?;
        self.map.insert(key, l);
        Ok(l)
    }
    pub fn destroy(&mut self, device: &ash::Device) {
        for (_, l) in self.map.drain() { unsafe { device.destroy_descriptor_set_layout(l, None) }; }
    }
}

pub struct PoolSizeRatio {
    pub ty: vk::DescriptorType,
    pub per_set: f32, 
}

pub struct DescriptorAllocator {
    ratios: Vec<PoolSizeRatio>,
    flags: vk::DescriptorPoolCreateFlags,
    ready: Vec<vk::DescriptorPool>,
    full: Vec<vk::DescriptorPool>,
    sets_per_pool: u32,
}

const MAX_SETS_PER_POOL: u32 = 4096;

impl DescriptorAllocator {
    pub fn new(ratios: Vec<PoolSizeRatio>, initial_sets: u32,
               flags: vk::DescriptorPoolCreateFlags) -> Self {
        Self { ratios, flags, ready: vec![], full: vec![], sets_per_pool: initial_sets }
    }

    fn create_pool(&self, device: &ash::Device) -> Result<vk::DescriptorPool> {
        let sizes: Vec<_> = self.ratios.iter().map(|r| vk::DescriptorPoolSize {
            ty: r.ty,
            descriptor_count: ((r.per_set * self.sets_per_pool as f32).ceil() as u32).max(1),
        }).collect();
        let info = vk::DescriptorPoolCreateInfo::default()
            .flags(self.flags)
            .max_sets(self.sets_per_pool)
            .pool_sizes(&sizes);
        Ok(unsafe { device.create_descriptor_pool(&info, None)? })
    }

    fn get_pool(&mut self, device: &ash::Device) -> Result<vk::DescriptorPool> {
        if let Some(p) = self.ready.pop() { return Ok(p); }
        let p = self.create_pool(device)?;
        self.sets_per_pool = (self.sets_per_pool * 3 / 2).min(MAX_SETS_PER_POOL);
        Ok(p)
    }

    pub fn allocate(
        &mut self,
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
        variable_count: Option<u32>,
    ) -> Result<vk::DescriptorSet> {
        let layouts = [layout];
        let counts = [variable_count.unwrap_or(0)];

        loop {
            let pool = self.get_pool(device)?;
            let mut var = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                .descriptor_counts(&counts);
            let mut info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            if variable_count.is_some() {
                info = info.push_next(&mut var);
            }

            match unsafe { device.allocate_descriptor_sets(&info) } {
                Ok(sets) => { self.ready.push(pool); return Ok(sets[0]); }
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                    self.full.push(pool); // retry with a fresh pool
                }
                Err(e) => { self.ready.push(pool); return Err(e.into()); }
            }
        }
    }

    pub fn reset(&mut self, device: &ash::Device) -> Result<()> {
        for p in self.ready.iter().chain(&self.full) {
            unsafe { device.reset_descriptor_pool(*p, vk::DescriptorPoolResetFlags::empty())? };
        }
        self.ready.append(&mut self.full);
        Ok(())
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        for p in self.ready.drain(..).chain(self.full.drain(..)) {
            unsafe { device.destroy_descriptor_pool(p, None) };
        }
    }
}

enum Info { Buffer(usize), Image(usize) }

#[derive(Default)]
pub struct DescriptorWriter {
    buffers: Vec<vk::DescriptorBufferInfo>,
    images: Vec<vk::DescriptorImageInfo>,
    writes: Vec<(u32, u32, vk::DescriptorType, Info)>,
}

impl DescriptorWriter {
    pub fn buffer(&mut self, binding: u32, ty: vk::DescriptorType,
                  buf: vk::Buffer, offset: u64, range: u64) -> &mut Self {
        self.buffers.push(vk::DescriptorBufferInfo { buffer: buf, offset, range });
        self.writes.push((binding, 0, ty, Info::Buffer(self.buffers.len() - 1)));
        self
    }

    pub fn image(&mut self, binding: u32, array_elem: u32, ty: vk::DescriptorType,
                 view: vk::ImageView, sampler: vk::Sampler, layout: vk::ImageLayout) -> &mut Self {
        self.images.push(vk::DescriptorImageInfo { sampler, image_view: view, image_layout: layout });
        self.writes.push((binding, array_elem, ty, Info::Image(self.images.len() - 1)));
        self
    }

    pub fn update(&self, device: &ash::Device, set: vk::DescriptorSet) {
        let writes: Vec<_> = self.writes.iter().map(|&(binding, elem, ty, ref info)| {
            let w = vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(binding).dst_array_element(elem).descriptor_type(ty);
            match *info {
                Info::Buffer(i) => w.buffer_info(std::slice::from_ref(&self.buffers[i])),
                Info::Image(i)  => w.image_info(std::slice::from_ref(&self.images[i])),
            }
        }).collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
}
