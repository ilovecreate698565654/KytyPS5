#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/profiler.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_bindless_table(graphics, m_command_scheduler) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_texture_cache.on_bindless_unregister = [this](ImageId id) {
		m_bindless_table.OnImageUnregistered(id);
	};
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
	// m_bindless_table is destroyed before m_texture_cache, whose destructor unregisters every
	// image; a pinned one would call back into the destroyed table.
	m_texture_cache.on_bindless_unregister = nullptr;
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		// Label completions on the priority thread still reference the GuestGpu.
		m_command_scheduler.DrainPriorityOperations();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::CanServeCleanRead(uint64_t fault_vaddr, uint64_t vaddr,
                                      uint64_t size) const noexcept {
	return IsMapped(vaddr, size) && m_page_manager.IsReadWatched(fault_vaddr) &&
	       m_buffer_cache.IsCleanForConcurrentRead(vaddr, size);
}

bool RenderContext::WriteFaultingStore(uint64_t fault_vaddr, uint64_t vaddr, const void* data,
                                       uint64_t size) noexcept {
	if (!IsMapped(vaddr, size) || !m_page_manager.IsReadWatched(fault_vaddr) || m_gpu == nullptr ||
	    m_gpu->IsStopping() || GuestGpu::IsGpuThread() || CommandScheduler::InDeferredOperation()) {
		return false;
	}
	bool written = false;
	m_gpu->SendCommandSync([this, vaddr, data, size, &written] {
		written = m_buffer_cache.WriteStore(vaddr, data, size);
	});
	return written;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		KYTY_PROFILER_BLOCK("RenderContext::UnmapMemory");
		// Completions write guest memory only for GPU-dirty buffer ranges, pending image
		// downloads and deferred labels; uploads copy guest memory when recorded and deleted
		// resources are retired by tick. Without any of those in the range, nothing asynchronous
		// touches it, so the unmap doesn't need to drain the GPU (a drain per guest munmap).
		const bool gpu_owned = m_buffer_cache.IsRegionGpuModified(vaddr, size) ||
		                       m_texture_cache.IsRegionGpuModified(vaddr, size) ||
		                       m_texture_cache.HasPendingDownload(vaddr, size) ||
		                       GuestGpu::LabelsDeferred();
		if (gpu_owned && m_command_scheduler.Active()) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::CacheDmaBases(const ShaderStageRuntime& runtime) {
	const auto&      program   = *runtime.program;
	const auto&      user_data = runtime.resources->user_data;
	std::shared_lock lock(m_mapped_ranges_mutex);
	for (const auto reg: program.info.dma_base_registers) {
		const auto index = static_cast<uint64_t>(reg) - program.user_data_base;
		if (reg < program.user_data_base || index + 1u >= user_data.size()) {
			continue;
		}
		const auto base = user_data[index] | (static_cast<uint64_t>(user_data[index + 1u]) << 32u);
		// The registers of an access that never runs may hold stale data. Mapped memory never
		// reaches the top of the address space, so a window that would overflow is skipped.
		if (base > UINT64_MAX - BufferCache::CACHING_PAGESIZE) {
			continue;
		}
		// DMA reaches only memory with a cached buffer; any other access records a fault and
		// reads zero, and the buffer arrives after the shader has run. That loses data the guest
		// writes for a single dispatch, such as the glyph bitmaps GTA V copies into its font atlas.
		m_mapped_ranges.ForEachInRange(base, BufferCache::CACHING_PAGESIZE,
		                               [this](uint64_t start, uint64_t end) {
			                               (void)m_buffer_cache.FindBuffer(start, end - start);
		                               });
	}
}

void RenderContext::PrepareBda() {
	const auto epoch = g_cpu_dirty_epoch.load(std::memory_order_acquire);
	if (epoch != m_bda_synced_epoch) {
		// The guest writes somewhere nearly all the time, so the epoch moves between most
		// dispatches; walk only the regions holding CPU-dirty pages, not every buffer.
		std::shared_lock lock(m_mapped_ranges_mutex);
		m_mapped_ranges.ForEach([this](uint64_t start, uint64_t end) {
			m_buffer_cache.SynchronizeCpuDirtyBuffersInRange(start, end - start);
		});
		m_bda_synced_epoch = epoch;
	}
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
	// The scan dispatches over the whole fault bitmap and may wait on older work; it ran after
	// every completed submission slice (hundreds per frame). Once per presented frame is enough
	// to create buffers for first-touched pages; dispatch recovery collects its own faults.
	const auto frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	if (m_fault_process_pending && frame != m_fault_process_frame) {
		m_fault_process_pending = false;
		m_fault_process_frame   = frame;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
