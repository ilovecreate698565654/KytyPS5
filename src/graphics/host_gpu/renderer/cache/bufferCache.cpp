#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
		                chunk, vk::AccessFlagBits::eHostWrite);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		// PrepareBda only visits existing tracker regions, so a new buffer's region must exist
		// (CPU-dirty) before the epoch bump makes it look synchronized.
		m_memory_tracker.TrackRange(buffer.CpuAddress(), buffer.Size());
		g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
		buffer.lru_id = m_lru_cache.Insert(id, LruClock());
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, LruClock());
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	// One reservation cannot exceed the download ring, so a larger range goes in ring-sized
	// windows; a window that does not fit drains the ring before it is mapped.
	const auto capacity = m_download_buffer.Size();
	bool       any      = false;
	for (uint64_t offset = 0; offset < size; offset += capacity) {
		any |= DownloadBufferWindow(buffer, vaddr + offset, std::min(capacity, size - offset));
	}
	return any;
}

bool BufferCache::DownloadBufferWindow(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    std::unique_lock lock(m_dirty_ranges_mutex);
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
			    m_downloading_ranges.Add(start, end - start);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    m_gpu_written_ranges.Subtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	const auto capacity = m_download_buffer.Size();
	for (size_t first = 0; first < copies.size();) {
		const auto base       = copies[first].dstOffset;
		auto       last       = first;
		uint64_t   batch_size = 0;
		while (last < copies.size()) {
			const auto end = copies[last].dstOffset - base + Common::AlignUp(copies[last].size, 64);
			if (end > capacity) {
				break;
			}
			batch_size = end;
			last++;
		}
		EXIT_IF(last == first);
		std::vector<vk::BufferCopy> batch(copies.begin() + static_cast<std::ptrdiff_t>(first),
		                                  copies.begin() + static_cast<std::ptrdiff_t>(last));
		for (auto& copy: batch) {
			copy.dstOffset -= base;
		}
		DownloadBufferCopies(buffer, std::move(batch), batch_size);
		first = last;
	}
	return true;
}

namespace {

// Research (KYTY_READBACK_STATS): of the GPU-dirty bytes downloaded, how many differ from what
// guest memory already held. Mostly unchanged means the dirty marking is too coarse (bindings
// marked written that the shaders never touched); mostly changed means genuine readbacks.
std::atomic<uint64_t> g_downloaded_bytes {0};
std::atomic<uint64_t> g_changed_bytes {0};

bool DownloadDiffEnabled() {
	static const bool enabled = std::getenv("KYTY_READBACK_STATS") != nullptr;
	return enabled;
}

void CountChangedBytes(uint64_t vaddr, const uint8_t* data, uint64_t size) {
	std::vector<uint8_t> current(size);
	if (!Libs::LibKernel::Memory::TryReadBacking(vaddr, current.data(), size)) {
		return;
	}
	uint64_t changed = 0;
	for (uint64_t i = 0; i < size; i++) {
		changed += current[i] != data[i] ? 1u : 0u;
	}
	g_downloaded_bytes.fetch_add(size, std::memory_order_relaxed);
	g_changed_bytes.fetch_add(changed, std::memory_order_relaxed);
}

} // namespace

void BufferCache::DownloadBufferCopies(Buffer& buffer, std::vector<vk::BufferCopy> copies,
                                       uint64_t total_size) {
	const auto buffer_address = buffer.CpuAddress();
	auto [mapped, offset]     = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	m_scheduler.DeferPriorityOperation([this, mapped, offset, total_size, buffer_address,
	                                    copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			if (DownloadDiffEnabled()) {
				CountChangedBytes(buffer_address + copy.srcOffset,
				                  mapped + (copy.dstOffset - offset), copy.size);
			}
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& copy: copies) {
			m_downloading_ranges.Subtract(buffer_address + copy.srcOffset, copy.size);
		}
	});
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	// Same test as the shader side: a value starting with 1 turns reports on.
	if (const char* reports = std::getenv("KYTY_WRITE_REPORTS");
	    reports != nullptr && reports[0] == '1') {
		// Host-visible so the retire reads the bits and clears them without GPU commands;
		// device-preferred so the shaders' atomics stay in video memory where possible.
		constexpr uint64_t RingBytes = 4ull * 1024 * 1024;
		m_report_ring                = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::Stream, 0,
		    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
		        vk::BufferUsageFlagBits::eShaderDeviceAddress,
		    RingBytes);
		EXIT_IF(m_report_ring->Mapped().empty() || !m_report_ring->HasDeviceAddress());
		m_report_ring_words = static_cast<uint32_t>(RingBytes / sizeof(uint32_t));
		std::memset(m_report_ring->Mapped().data(), 0, RingBytes);
		m_report_ring->Flush(0, RingBytes);
		m_scheduler.SetPreSubmitHook([this] { FlushWriteReports(); });
	}
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (m_report_ring != nullptr) {
		m_scheduler.SetPreSubmitHook(nullptr);
	}
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

bool BufferCache::WriteClean(uint64_t vaddr, const void* data, uint64_t size) {
	if (!GuestGpu::IsGpuThread() || size == 0 || !GuestRange {vaddr, size}.Valid() ||
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	{
		// A download publishes the GPU's older value of these bytes when it lands.
		std::shared_lock lock(m_dirty_ranges_mutex);
		if (m_downloading_ranges.Intersects(vaddr, size)) {
			return false;
		}
	}

	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || !m_slot_buffers[*owner].IsInBounds(vaddr, size) ||
	    !Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	WriteDataBuffer(m_slot_buffers[*owner], vaddr, data, size);
	if (HasGpuDirtyBytes(vaddr, size)) {
		// Overwritten in full: guest memory holds the value the GPU copy will have.
		std::unique_lock lock(m_dirty_ranges_mutex);
		m_gpu_modified_ranges.Subtract(vaddr, size);
		m_gpu_written_ranges.Subtract(vaddr, size);
	}
	// A tracker page marked GPU-modified must still hold GPU-dirty bytes; otherwise the GC's
	// download of an aged buffer finds nothing to copy and exits.
	for (auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	     page < Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE); page += TRACKER_PAGE_SIZE) {
		if (!HasGpuDirtyBytes(page, TRACKER_PAGE_SIZE)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
		}
	}
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

namespace {

// Research: KYTY_READBACK_STATS=1 prints, every 2 s, how guest readbacks drain the GPU: how many,
// how many distinct windows, writes vs reads, how many windows were already downloaded as part of
// an earlier drain's batch, and the most frequent windows. GPU thread only.
struct ReadbackStats {
	uint64_t                                calls        = 0;
	uint64_t                                stores       = 0;
	uint64_t                                report_written_pages = 0;
	uint64_t                                report_cleaned_pages = 0;
	uint64_t                                writes       = 0;
	uint64_t                                fast_path    = 0;
	uint64_t                                drains       = 0;
	uint64_t                                was_hot      = 0;
	uint64_t                                extras       = 0;
	uint64_t                                extra_bytes  = 0;
	std::unordered_map<uint64_t, uint32_t>  windows;
	std::chrono::steady_clock::time_point   since        = std::chrono::steady_clock::now();
	uint32_t                                reports      = 0;

	static bool Enabled() {
		static const bool enabled = std::getenv("KYTY_READBACK_STATS") != nullptr;
		return enabled;
	}
	void Report() {
		const auto now = std::chrono::steady_clock::now();
		if (now - since < std::chrono::seconds(2) || reports >= 30) {
			return;
		}
		std::vector<std::pair<uint64_t, uint32_t>> top(windows.begin(), windows.end());
		std::sort(top.begin(), top.end(),
		          [](const auto& a, const auto& b) { return a.second > b.second; });
		std::string text = fmt::format(
		    "Readback stats (2 s): calls={} writes={} merged_stores={} fast={} drains={} "
		    "distinct_windows={} already_batched={} extra_downloads={} ({} KiB)\n",
		    calls, writes, stores, fast_path, drains, windows.size(), was_hot, extras, extra_bytes / 1024u);
		const auto downloaded = g_downloaded_bytes.exchange(0, std::memory_order_relaxed);
		const auto changed    = g_changed_bytes.exchange(0, std::memory_order_relaxed);
		text += fmt::format("  downloaded {} KiB of GPU-dirty bytes, {} KiB changed ({}%)\n",
		                    downloaded / 1024u, changed / 1024u,
		                    downloaded != 0 ? changed * 100u / downloaded : 0u);
		if (report_written_pages + report_cleaned_pages != 0) {
			text += fmt::format("  write reports: {} pages written, {} pages cleaned\n",
			                    report_written_pages, report_cleaned_pages);
		}
		for (size_t i = 0; i < std::min<size_t>(top.size(), 8u); i++) {
			text += fmt::format("  window 0x{:012x} x{}\n", top[i].first, top[i].second);
		}
		Log::WriteToConsoleAndLog(text);
		const auto reported = reports;
		*this   = {};
		since   = now;
		reports = reported + 1u;
	}
};

ReadbackStats& GetReadbackStats() {
	static ReadbackStats stats;
	return stats;
}

} // namespace

uint64_t BufferCache::WriteReportRingAddress() const noexcept {
	return m_report_ring != nullptr ? m_report_ring->BufferDeviceAddress() : 0;
}

bool BufferCache::AllocateReportWords(uint32_t words, uint32_t& offset, uint32_t& alloc) {
	uint32_t used = 0;
	for (const auto& a: m_report_allocs) {
		used += a.words;
	}
	auto start = m_report_ring_head;
	auto taken = words;
	if (start + words > m_report_ring_words) {
		taken += m_report_ring_words - start; // skip the tail; allocations stay contiguous
		start = 0;
	}
	if (used + taken > m_report_ring_words) {
		return false;
	}
	m_report_allocs.push_back({start, taken, false});
	alloc              = m_report_alloc_base + static_cast<uint32_t>(m_report_allocs.size() - 1);
	offset             = start;
	m_report_ring_head = start + words;
	return true;
}

void BufferCache::FreeReportWords(uint32_t alloc) {
	m_report_allocs[alloc - m_report_alloc_base].freed = true;
	while (!m_report_allocs.empty() && m_report_allocs.front().freed) {
		m_report_allocs.pop_front();
		m_report_alloc_base++;
	}
}

void BufferCache::ConfirmWritten(uint64_t vaddr, uint64_t size) {
	std::unique_lock lock(m_dirty_ranges_mutex);
	m_gpu_modified_ranges.ForEachInRange(
	    vaddr, size, [&](uint64_t begin, uint64_t end) { m_gpu_written_ranges.Add(begin, end - begin); });
}

uint32_t BufferCache::OpenWriteReport(uint64_t vaddr, uint64_t size) {
	constexpr uint32_t Off = 0xffffffffu;
	if (m_report_ring == nullptr || size == 0) {
		return Off;
	}
	const auto phase_dwords = static_cast<uint32_t>((vaddr & (TRACKER_PAGE_SIZE - 1)) / 4u);
	const auto dwords       = (size + 3u) / 4u;
	const auto pages        = (phase_dwords + dwords + 1023u) / 1024u;
	const auto words        = (pages + 31u) / 32u;
	const auto key          = vaddr * 0x9e3779b97f4a7c15ull ^ size;
	if (const auto found = m_current_report_index.find(key); found != m_current_report_index.end()) {
		auto& report = m_current_reports[found->second];
		if (report.vaddr == vaddr && report.size == size) {
			report.open = true; // bound again before this tick is submitted
			return (report.word << 10u) | phase_dwords;
		}
	}
	uint32_t offset = 0;
	uint32_t alloc  = 0;
	if (words >= (1u << 22u) || !AllocateReportWords(static_cast<uint32_t>(words), offset, alloc) ||
	    offset >= (1u << 22u) - 1u) {
		ConfirmWritten(vaddr, size);
		return Off;
	}
	m_current_report_index[key] = m_current_reports.size();
	m_current_reports.push_back({.vaddr = vaddr,
	                             .size  = size,
	                             .word  = offset,
	                             .words = static_cast<uint32_t>(words),
	                             .alloc = alloc,
	                             .open  = true});
	return (offset << 10u) | phase_dwords;
}

void BufferCache::CommitWriteReports() {
	for (auto& report: m_current_reports) {
		report.open = false;
	}
}

void BufferCache::FlushWriteReports() {
	if (m_current_reports.empty()) {
		return;
	}
	// Reports still open belong to a draw or dispatch not recorded yet (a stream-buffer wrap can
	// submit while one is being prepared): they move on with it to the next tick.
	std::vector<WriteReport> retiring;
	std::vector<WriteReport> kept;
	for (auto& report: m_current_reports) {
		(report.open ? kept : retiring).push_back(report);
	}
	m_current_reports = std::move(kept);
	m_current_report_index.clear();
	for (size_t i = 0; i < m_current_reports.size(); i++) {
		const auto& report = m_current_reports[i];
		m_current_report_index[report.vaddr * 0x9e3779b97f4a7c15ull ^ report.size] = i;
	}
	if (retiring.empty()) {
		return;
	}
	auto& command = m_scheduler.Current();
	command.EndRendering();
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                 vk::PipelineStageFlagBits::eHost, {}, 1, &barrier, 0, nullptr,
	                                 0, nullptr);
	m_retiring_ranges.push_back(retiring);
	m_scheduler.DeferOperation([this, reports = std::move(retiring)]() mutable {
		RetireWriteReports(std::move(reports));
	});
}

void BufferCache::RetireWriteReports(std::vector<WriteReport> reports) {
	KYTY_PROFILER_BLOCK("BufferCache::RetireWriteReports");
	EXIT_IF(m_retiring_ranges.empty());
	m_retiring_ranges.pop_front(); // batches complete in submission order
	auto* ring = reinterpret_cast<uint32_t*>(m_report_ring->Mapped().data());
	for (const auto& report: reports) {
		m_report_ring->Invalidate(report.word * sizeof(uint32_t), report.words * sizeof(uint32_t));
	}
	const auto page_segment = [](const WriteReport& report, uint64_t page, uint64_t& begin,
	                             uint64_t& end) {
		const auto base = Common::AlignDown(report.vaddr, TRACKER_PAGE_SIZE);
		begin           = std::max(report.vaddr, base + page * TRACKER_PAGE_SIZE);
		end             = std::min(report.vaddr + report.size, base + (page + 1u) * TRACKER_PAGE_SIZE);
		return begin < end;
	};
	const auto page_count = [](const WriteReport& report) {
		const auto base = Common::AlignDown(report.vaddr, TRACKER_PAGE_SIZE);
		return (report.vaddr + report.size - base + TRACKER_PAGE_SIZE - 1u) / TRACKER_PAGE_SIZE;
	};
	uint64_t written_pages = 0;
	uint64_t cleaned_pages = 0;
	// Written pages first, so another report of this batch can't clean them.
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& report: reports) {
			const auto pages = page_count(report);
			for (uint64_t page = 0; page < pages; page++) {
				if ((ring[report.word + page / 32u] & (1u << (page % 32u))) == 0) {
					continue;
				}
				uint64_t begin = 0;
				uint64_t end   = 0;
				if (page_segment(report, page, begin, end)) {
					written_pages++;
					m_gpu_modified_ranges.ForEachInRange(begin, end - begin, [&](uint64_t b, uint64_t e) {
						m_gpu_written_ranges.Add(b, e - b);
					});
				}
			}
		}
	}
	// A page can only be cleaned when no other report still in flight or being recorded covers
	// it: that work may yet write it.
	RangeSet pending;
	for (const auto& batch: m_retiring_ranges) {
		for (const auto& report: batch) {
			pending.Add(report.vaddr, report.size);
		}
	}
	for (const auto& report: m_current_reports) {
		pending.Add(report.vaddr, report.size);
	}
	std::vector<uint64_t> touched;
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& report: reports) {
			const auto pages = page_count(report);
			for (uint64_t page = 0; page < pages; page++) {
				if ((ring[report.word + page / 32u] & (1u << (page % 32u))) != 0) {
					continue;
				}
				uint64_t begin = 0;
				uint64_t end   = 0;
				if (!page_segment(report, page, begin, end) ||
				    !m_gpu_modified_ranges.Intersects(begin, end - begin) ||
				    pending.Intersects(begin, end - begin) ||
				    m_downloading_ranges.Intersects(begin, end - begin) ||
				    m_gpu_written_ranges.Intersects(begin, end - begin)) {
					continue;
				}
				m_gpu_modified_ranges.Subtract(begin, end - begin);
				touched.push_back(Common::AlignDown(begin, TRACKER_PAGE_SIZE));
				cleaned_pages++;
			}
		}
	}
	for (const auto page: touched) {
		if (m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE) &&
		    !HasGpuDirtyBytes(page, TRACKER_PAGE_SIZE)) {
			bool downloading = false;
			{
				std::shared_lock lock(m_dirty_ranges_mutex);
				downloading = m_downloading_ranges.Intersects(page, TRACKER_PAGE_SIZE);
			}
			if (!downloading) {
				m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
			}
		}
	}
	for (const auto& report: reports) {
		std::memset(ring + report.word, 0, report.words * sizeof(uint32_t));
		m_report_ring->Flush(report.word * sizeof(uint32_t), report.words * sizeof(uint32_t));
		FreeReportWords(report.alloc);
	}
	if (ReadbackStats::Enabled()) {
		auto& stats = GetReadbackStats();
		stats.report_written_pages += written_pages;
		stats.report_cleaned_pages += cleaned_pages;
	}
}

bool BufferCache::WriteStore(uint64_t vaddr, const void* data, uint64_t size) {
	constexpr uint32_t StoresPerPagePerFrame = 16;
	const auto         frame =
	    m_scheduler.Context().GetGraphics().presented_frames.load(std::memory_order_relaxed);
	if (frame != m_store_budget_frame) {
		m_store_budget.clear();
		m_store_budget_frame = frame;
	}
	auto& used = m_store_budget[vaddr / TRACKER_PAGE_SIZE];
	if (used >= StoresPerPagePerFrame || !WriteClean(vaddr, data, size)) {
		return false;
	}
	used++;
	if (ReadbackStats::Enabled()) {
		auto& stats = GetReadbackStats();
		stats.stores++;
		stats.Report();
	}
	return true;
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// A guest thread's readback used to hold the GPU command thread for the whole wait (65% of it
	// in Wolverine's forest): it now only records the download and submits, and the faulting
	// thread waits for that submission by itself. KYTY_SYNC_READBACK=1 restores the old wait.
	static const bool async_enabled = std::getenv("KYTY_SYNC_READBACK") == nullptr;
	const bool        async_drain   = async_enabled && !GuestGpu::IsGpuThread();
	uint64_t          drain_tick    = 0;
	uint64_t          drain_begin   = 0;
	uint64_t          drain_end     = 0;
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write, async_drain,
	                                                &drain_tick, &drain_begin, &drain_end] {
		KYTY_PROFILER_BLOCK("BufferCache::ReadMemory");
		auto*      stats = ReadbackStats::Enabled() ? &GetReadbackStats() : nullptr;
		if (stats != nullptr) {
			stats->calls++;
			stats->writes += is_write ? 1u : 0u;
			stats->Report();
		}
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

		// The page is protected as GPU-written, but none of its bytes are waiting for a download:
		// the bytes the GPU wrote are elsewhere in the window, or were downloaded with another
		// page. The page is current, so lift its protection. Downloading the window instead drained
		// the GPU (~30 ms a fault) for guest reads of structs next to the command processor's
		// labels.
		const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
		if (m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin) &&
		    !HasGpuDirtyBytes(page_begin, page_end - page_begin)) {
			bool downloading = false;
			{
				std::shared_lock lock(m_dirty_ranges_mutex);
				downloading = m_downloading_ranges.Intersects(page_begin, page_end - page_begin);
			}
			if (downloading) {
				// Another readback's download of these bytes has not landed yet: lifting the
				// protection now would let the guest read the old value.
				if (async_drain) {
					drain_tick  = m_scheduler.CurrentTick();
					drain_begin = page_begin;
					drain_end   = page_end;
					m_scheduler.Flush();
					return;
				}
				const auto tick = m_scheduler.CurrentTick();
				m_scheduler.Wait(tick);
				m_scheduler.WaitPriorityOperations(tick);
			}
			m_memory_tracker.UnmarkRegionAsGpuModified(page_begin, page_end - page_begin);
			if (is_write) {
				m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			}
			if (stats != nullptr) {
				stats->fast_path++;
			}
			return;
		}

		// Widen nearby CPU reads so they share one GPU drain.
		constexpr uint64_t WindowSize   = 512 * 1024;
		const auto         buffer_begin = buffer.CpuAddress();
		const auto         buffer_end   = buffer_begin + buffer.Size();
		const auto window_begin = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
		const auto window_end = std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);

		// Windows the guest read recently tend to be read again every frame (Wolverine: ~200
		// drains per frame, one window each). A drain paid here also downloads the recent windows
		// the GPU has dirtied since, so their next reads don't each pay a drain of their own.
		const auto frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
		if (stats != nullptr) {
			stats->drains++;
			stats->windows[window_begin]++;
			stats->was_hot += std::any_of(m_hot_windows.begin(), m_hot_windows.end(),
			                              [&](const HotWindow& w) { return w.begin == window_begin; })
			                      ? 1u
			                      : 0u;
		}
		std::erase_if(m_hot_windows, [&](const HotWindow& window) {
			return window.frame + 2u < frame || window.begin == window_begin;
		});
		if (m_hot_windows.size() < MaxHotWindows) {
			m_hot_windows.push_back({window_begin, window_end, frame});
		}
		KYTY_PROFILER_BLOCK("BufferCache::ReadMemory drain");
		if (DownloadBufferMemory(buffer, window_begin, window_end - window_begin)) {
			if (async_drain) {
				drain_tick  = m_scheduler.CurrentTick();
				drain_begin = window_begin;
				drain_end   = window_end;
				m_scheduler.Flush();
				return;
			}
			// Measured on Wolverine (KYTY_READBACK_STATS): the drains cycle over 4 windows the GPU
			// re-dirties between reads, so batching only re-downloaded ~30 MB/s for nothing.
			// Opt-in (KYTY_READBACK_BATCH=1) until write tracking is precise.
			static const bool batch  = std::getenv("KYTY_READBACK_BATCH") != nullptr;
			const uint64_t    budget_limit = batch ? 8u * 1024u * 1024u : 0u;
			uint64_t          budget       = budget_limit;
			std::vector<std::pair<uint64_t, uint64_t>> extra;
			for (const auto& window: m_hot_windows) {
				const auto bytes = window.end - window.begin;
				if (window.begin == window_begin || bytes > budget) {
					continue;
				}
				const auto* owner = m_page_table.Find(window.begin >> PageTable::kPageBits);
				if (owner == nullptr || !*owner ||
				    !m_slot_buffers[*owner].IsInBounds(window.begin, bytes) ||
				    !m_memory_tracker.IsRegionGpuModified(window.begin, bytes)) {
					continue;
				}
				if (DownloadBufferMemory(m_slot_buffers[*owner], window.begin, bytes)) {
					extra.emplace_back(window.begin, bytes);
					budget -= bytes;
					if (stats != nullptr) {
						stats->extras++;
						stats->extra_bytes += bytes;
					}
				}
			}
			const auto tick = m_scheduler.CurrentTick();
			m_scheduler.Wait(tick);
			m_scheduler.WaitPriorityOperations(tick);
			m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
			for (const auto& [begin, bytes]: extra) {
				m_memory_tracker.UnmarkRegionAsGpuModified(begin, bytes);
			}
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
	});
	if (drain_tick == 0) {
		return;
	}
	{
		KYTY_PROFILER_BLOCK("BufferCache::ReadMemory wait");
		m_scheduler.WaitSubmitted(drain_tick);
		m_scheduler.WaitPriorityOperations(drain_tick);
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, drain_begin, drain_end, vaddr, size,
	                                                is_write] {
		FinishAsyncDrain(drain_begin, drain_end, vaddr, size, is_write);
	});
}

void BufferCache::FinishAsyncDrain(uint64_t begin, uint64_t end, uint64_t vaddr, uint64_t size,
                                   bool is_write) {
	KYTY_PROFILER_BLOCK("BufferCache::FinishAsyncDrain");
	// The GPU thread kept recording while the guest thread waited: pages it has marked written
	// again, or whose bytes another download is still bringing back, stay protected (the guest
	// faults again and starts a new readback).
	for (auto page = Common::AlignDown(begin, TRACKER_PAGE_SIZE);
	     page < Common::AlignUp(end, TRACKER_PAGE_SIZE); page += TRACKER_PAGE_SIZE) {
		if (!m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE) ||
		    HasGpuDirtyBytes(page, TRACKER_PAGE_SIZE)) {
			continue;
		}
		{
			std::shared_lock lock(m_dirty_ranges_mutex);
			if (m_downloading_ranges.Intersects(page, TRACKER_PAGE_SIZE)) {
				continue;
			}
		}
		m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
	}
	if (is_write) {
		const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
		if (!m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin)) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
	}
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	if (vaddr < CACHING_PAGESIZE) {
		// Guest page 0 is never mapped; a request here is a garbage or null descriptor that
		// slipped past the null checks. The memory tracker cannot hold address 0, so name the
		// caller now rather than in the garbage collector later.
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 8) {
			LOGF("BufferCache: buffer requested in guest page 0: vaddr=0x%016" PRIx64
			     " size=0x%016" PRIx64 "\n%s",
			     vaddr, size, Common::HostBacktrace().c_str());
		}
	}
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

// Reads guest memory for an upload. The guest can unmap memory a cached buffer still covers: AMM
// maps streaming textures in blocks, and a texture upload once read one byte past the end of the
// block it had mapped. Such bytes read as zero, like ObtainBufferForImage's direct path.
static void ReadGuestForUpload(uint8_t* destination, uint64_t address, uint64_t size) {
	if (!Libs::LibKernel::Memory::TryReadBacking(address, destination, size) &&
	    !Libs::LibKernel::Memory::TryReadSparseBacking(address, destination, size)) {
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1) < 16) {
			LOGF("BufferCache: upload of unmapped guest memory 0x%016" PRIx64 " size=0x%" PRIx64
			     " reads as zero\n",
			     address, size);
		}
		std::memset(destination, 0, static_cast<size_t>(size));
	}
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			ReadGuestForUpload(mapped + copy.srcOffset, address, copy.size);
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		ReadGuestForUpload(temporary->Mapped().data() + copy.srcOffset, address, copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id, bool needs_device_address,
                                                       bool reported) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (!is_written && size <= CACHING_PAGESIZE &&
	    (!needs_device_address || m_stream_buffer.HasDeviceAddress()) &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			std::memcpy(mapped, reinterpret_cast<const void*>(vaddr), size);
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		std::unique_lock lock(m_dirty_ranges_mutex);
		m_gpu_modified_ranges.Add(vaddr, size);
		if (WriteReportsEnabled() && !reported) {
			m_gpu_written_ranges.Add(vaddr, size);
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr) {
		EXIT("BufferCache: staging reservation failed for guest image\n");
	}
	if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, staging, size)) {
		std::memset(staging, 0, static_cast<size_t>(size));
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsCleanForConcurrentRead(uint64_t vaddr, uint64_t size) const {
	std::shared_lock lock(m_dirty_ranges_mutex);
	return !m_gpu_modified_ranges.Intersects(vaddr, size) &&
	       !m_downloading_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

uint64_t BufferCache::LruClock() const noexcept {
	return m_graphics.presented_frames.load(std::memory_order_relaxed) + m_gc_tick / 512;
}

void BufferCache::RunGarbageCollector() {
	m_gc_tick++;
	const auto clock = LruClock();
	// Pressure is judged by this cache's own bytes. Device-wide usage also counts the images
	// spilled to host memory, which on a 6 GB card keeps it above the critical mark forever
	// and has the collector destroy and recreate every buffer twice a second (a third of
	// the GPU thread's time in the world). KYTY_GC_DEVICE_BYTES=1 restores that policy.
	static const bool device_bytes = std::getenv("KYTY_GC_DEVICE_BYTES") != nullptr;
	if (device_bytes && m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	// Ages in frames, as in the texture cache: a buffer used this frame or the last is
	// never a candidate, whatever the submission count.
	const uint64_t age        = std::min<uint64_t>(aggressive ? 2 : 4, clock);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(clock - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		if (buffer.CpuAddress() == 0) {
			// See CreateBuffer: the tracker rejects address 0, so this one is never collected.
			return false;
		}
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		// An aged GPU-dirty buffer is downloaded and dropped under either policy. Retaining it
		// (the non-aggressive rule before) leaves its GPU-written bytes owning the pages
		// forever, and the images that share those pages are then re-sourced from the
		// buffer's stale contents: the world renders black. Measured 2026-09-21.
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		// A page can stay marked GPU-modified with no dirty bytes left to copy (the CPU overwrote
		// them); such a buffer has nothing to download and retires like a clean one.
		if (dirty && DownloadBufferMemory(buffer, buffer.CpuAddress(), buffer.Size())) {
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeCpuDirtyBuffersInRange(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.ForEachMaybeCpuDirtyRegion(
	    vaddr, size, [this](uint64_t address, uint64_t bytes) {
		    SynchronizeBuffersInRange(address, bytes);
	    });
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
