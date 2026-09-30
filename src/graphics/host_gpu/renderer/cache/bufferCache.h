#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	// GPU thread, after a guest thread waited out a readback it started: unprotects the pages of
	// [begin, end) that no longer hold GPU-dirty bytes or pending downloads.
	void                   FinishAsyncDrain(uint64_t begin, uint64_t end, uint64_t vaddr,
	                                        uint64_t size, bool is_write);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// Marks a buffer bound through FindBuffer as used, so the collector doesn't evict a buffer
	// that is bound every frame. GPU thread only (the LRU is not synchronized).
	void MarkUsed(BufferId id) {
		if (!IsBufferInvalid(id)) {
			TouchBuffer(m_slot_buffers[id]);
		}
	}
	// needs_device_address: the caller reads the data through a buffer device address, which
	// the stream buffer used for small CPU-written reads does not have.
	// reported: the caller binds a written range whose shader reports the pages it writes, and
	// calls OpenWriteReport for it next (see below); every other GPU write is taken as exact.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer      = false,
	                                                        BufferId id                   = {},
	                                                        bool     needs_device_address = false,
	                                                        bool     reported             = false);

	// Write reports (KYTY_WRITE_REPORTS=1; GPU thread only). A written storage binding is marked
	// GPU-dirty whole, although shaders usually write a small part of it (Wolverine: 3-17% of the
	// downloaded "dirty" bytes had changed). A reporting shader sets one bit per 4 KiB page it
	// writes in a ring; when its submission completes, pages whose bit stayed clear stop being
	// GPU-dirty, so guest reads of them no longer drain the GPU.
	[[nodiscard]] bool     WriteReportsEnabled() const noexcept { return m_report_ring != nullptr; }
	[[nodiscard]] uint64_t WriteReportRingAddress() const noexcept;
	// The bound range [vaddr, vaddr + size) of a binding obtained with reported = true. Returns
	// (word_base << 10) | phase_dwords: page p = (dword_index + phase_dwords) >> 10 relative to
	// AlignDown(vaddr, 4 KiB) sets bit (p & 31) of ring word word_base + (p >> 5). 0xFFFFFFFF
	// means reporting is off for the binding (the range is then taken as written in full).
	[[nodiscard]] uint32_t OpenWriteReport(uint64_t vaddr, uint64_t size);
	// After each draw or dispatch is recorded: its open reports belong to the current tick.
	void                   CommitWriteReports();
	// GPU thread, at each end-of-pipe label (KYTY_READBACK_PREFETCH=1): the windows the guest
	// read back recently are downloaded right behind the work the label releases, and unprotected
	// once they land, so the guest's next read of them runs without a fault or a drain.
	void                   PrefetchHotWindows();
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer*       GetGdsBuffer() noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool              IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool              HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// Any thread: none of the bytes is GPU-written, or on its way back from the GPU, so guest
	// memory holds their current value even when their page is protected.
	[[nodiscard]] bool              IsCleanForConcurrentRead(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool              IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool              IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void                            ProcessFaultBuffer();
	// GPU thread: writes bytes on a page protected because the GPU wrote to it, without
	// downloading the page: the host copy through the backing store, the GPU copy in the command
	// stream, after the GPU's earlier writes. Bytes the GPU had written are then current on both
	// sides and no longer need a download. False when there is nothing to save (the page is not
	// protected as GPU-written) or it would be wrong (a download of these bytes is in flight, or
	// no cached buffer covers them); the caller then writes normally.
	[[nodiscard]] bool              WriteClean(uint64_t vaddr, const void* data, uint64_t size);
	// GPU thread: WriteClean for a guest store that faulted. Each page takes a few per frame;
	// beyond that the caller downloads it once, which unprotects it for bulk writers.
	[[nodiscard]] bool              WriteStore(uint64_t vaddr, const void* data, uint64_t size);
	[[nodiscard]] ShaderFaultReport CollectFaults() { return m_fault_manager.CollectFaults(); }
	[[nodiscard]] uint64_t          UnattributedFaults() const noexcept {
		return m_fault_manager.UnattributedFaults();
	}
	void                            SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Same, but visits only the tracker regions that may hold CPU-dirty pages.
	void                            SynchronizeCpuDirtyBuffersInRange(uint64_t vaddr, uint64_t size);
	void                            RunGarbageCollector();

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void                   Unregister(BufferId id);
	template <bool insert>
	void                     ChangeRegister(BufferId id);
	void                     DeleteBuffer(BufferId id);
	[[nodiscard]] bool       SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                           bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Queues backing publication; callers wait before clearing dirty pages or reusing their data.
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool DownloadBufferWindow(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               DownloadBufferCopies(Buffer& buffer, std::vector<vk::BufferCopy> copies,
	                                        uint64_t total_size);

	struct WriteReport {
		uint64_t vaddr     = 0;
		uint64_t size      = 0;
		uint32_t word      = 0; // first ring word
		uint32_t words     = 0;
		uint32_t alloc     = 0; // index into m_report_allocs
		bool     open      = true;
	};
	struct ReportAlloc {
		uint32_t offset = 0;
		uint32_t words  = 0; // including words skipped at the ring's end
		bool     freed  = false;
	};
	[[nodiscard]] bool AllocateReportWords(uint32_t words, uint32_t& offset, uint32_t& alloc);
	void               FreeReportWords(uint32_t alloc);
	void               FlushWriteReports();
	void               RetireWriteReports(std::vector<WriteReport> reports);
	void               ConfirmWritten(uint64_t vaddr, uint64_t size);
	[[nodiscard]] static bool PrefetchEnabled();

	// Adaptive readback precision (KYTY_READBACK_ADAPTIVE=1). A window whose downloads keep
	// bringing back the bytes guest memory already held becomes trusted: its read faults then
	// unprotect its pages without a download, and only every AdaptiveVerifyFaults-th fault (or
	// after AdaptiveVerifyFrames frames) pays a verified download, which revokes the trust if a
	// single byte differed. See TrustedReadback for what that costs in correctness.
	struct AdaptiveWindow {
		uint32_t unchanged    = 0; // consecutive downloads that changed no byte
		uint32_t faults       = 0; // trusted faults since the last verified download
		uint64_t verify_frame = 0; // frame of the last verified download
	};
	static constexpr uint32_t AdaptiveTrustDownloads = 8;
	static constexpr uint32_t AdaptiveVerifyFaults   = 16;
	static constexpr uint64_t AdaptiveVerifyFrames   = 30;
	static constexpr size_t   MaxAdaptiveWindows     = 256;
	[[nodiscard]] static bool AdaptiveEnabled();
	// Download completion (any thread): whether each window the copies touch changed.
	void RecordAdaptiveDownload(uint64_t buffer_address, const std::vector<vk::BufferCopy>& copies,
	                            const uint8_t* mapped, uint64_t offset);
	// GPU thread: true when the fault on [page_begin, page_end) was served from guest memory.
	[[nodiscard]] bool TrustedReadback(uint64_t window_begin, uint64_t window_end, uint64_t page_begin,
	                                   uint64_t page_end, uint64_t frame);

	GraphicContext&                                    m_graphics;
	// Guest-read windows of the last frames, downloaded together when one read drains (GPU thread).
	struct HotWindow {
		uint64_t begin;
		uint64_t end;
		uint64_t frame;
	};
	static constexpr size_t                            MaxHotWindows = 64;
	std::vector<HotWindow>                             m_hot_windows;
	// Adaptive windows by window_begin; download completions update them off the GPU thread.
	std::mutex                                         m_adaptive_mutex;
	std::unordered_map<uint64_t, AdaptiveWindow>       m_adaptive_windows;
	std::unordered_map<uint64_t, uint32_t>             m_store_budget; // page -> stores this frame
	uint64_t                                           m_store_budget_frame = ~0ull;
	CommandScheduler&                                  m_scheduler;
	FaultManager                                       m_fault_manager;
	Buffer                                             m_gds_buffer;
	Buffer                                             m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                         m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                          m_buffers;
	PageTable                                          m_page_table;
	RangeSet                                           m_gpu_modified_ranges;
	// Write reports: GPU-dirty bytes known to be written (unreported writes, and reported pages
	// whose bit was set). A report never cleans these; they leave with the dirty bytes.
	RangeSet                                           m_gpu_written_ranges;
	std::unique_ptr<Buffer>                            m_report_ring;
	uint32_t                                           m_report_ring_words = 0;
	uint32_t                                           m_report_ring_head  = 0;
	uint32_t                                           m_report_alloc_base = 0; // index of front
	std::deque<ReportAlloc>                            m_report_allocs;
	std::vector<WriteReport>                           m_current_reports; // this tick's reports
	std::unordered_map<uint64_t, size_t>               m_current_report_index;
	std::deque<std::vector<WriteReport>>               m_retiring_ranges;
	// Bytes whose download is recorded but not yet in guest memory.
	RangeSet                                           m_downloading_ranges;
	// Guards changes to both range sets (GPU thread and download completions) against
	// IsCleanForConcurrentRead; the GPU thread reads them without it.
	mutable std::shared_mutex                          m_dirty_ranges_mutex;
	MemoryTracker                                      m_memory_tracker;
	StreamBuffer                                       m_staging_buffer;
	StreamBuffer                                       m_stream_buffer;
	StreamBuffer                                       m_download_buffer;
	StreamBuffer                                       m_device_buffer;
	TextureCache&                                      m_texture_cache;
	uint64_t                                           m_total_used_memory = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	[[nodiscard]] uint64_t LruClock() const noexcept;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
