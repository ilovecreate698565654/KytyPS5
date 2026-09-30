#include "graphics/host_gpu/renderer/pipeline/shaderDiskCache.h"

#include "common/file.h"
#include "common/logging/log.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <xxhash.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
extern char** environ; // NOLINT(readability-redundant-declaration)
#endif

namespace Libs::Graphics {

namespace IR = ShaderRecompiler::IR;

namespace {

// Bump when the file layout or the serialized structures change meaning. Translation changes
// need no bump: every rebuild of the emulator starts the file over (see BuildId).
constexpr uint32_t FormatVersion = 1;
constexpr uint64_t FileMagic     = 0x31565053594b594bull; // "KYKYSPV1"
constexpr uint32_t RecordMagic   = 0x4345524bu;           // "KREC"
constexpr uint32_t RecordPlan        = 1;
constexpr uint32_t RecordPermutation = 2;
constexpr uint64_t MaxFileSize       = 1ull << 31u;
constexpr uint32_t MaxPayloadSize    = 64u << 20u;
constexpr uint32_t SpirvMagic        = 0x07230203u;

#pragma pack(push, 1)
struct FileHeader {
	uint64_t magic       = FileMagic;
	uint32_t version     = FormatVersion;
	uint32_t header_size = sizeof(FileHeader);
	uint64_t build_low   = 0;
	uint64_t build_high  = 0;
};

struct RecordHeader {
	uint32_t magic    = RecordMagic;
	uint32_t kind     = 0;
	uint64_t key_low  = 0;
	uint64_t key_high = 0;
	uint32_t size     = 0;
	uint32_t reserved = 0;
	uint64_t hash     = 0;
};
#pragma pack(pop)

static_assert(sizeof(FileHeader) == 32);
static_assert(sizeof(RecordHeader) == 40);

// The structures below are written field by field. When one of these sizes changes, a field
// was added or removed: update the matching Write/Read pair, then the size (Windows x64 sizes).
// A field that fits into padding goes unnoticed here; for shader info and binding layouts the
// round-trip comparison in Store* still refuses to keep what does not read back equal.
#if defined(_WIN64)
static_assert(sizeof(IR::ResourcePlan) == 648);
static_assert(sizeof(IR::ShaderInfo) == 216);
static_assert(sizeof(IR::ImageResource) == 88);
static_assert(sizeof(IR::StageInput) == 56);
static_assert(sizeof(IR::StageOutput) == 48);
static_assert(sizeof(IR::DescriptorSource) == 216);
static_assert(sizeof(IR::DescriptorSource::IndirectImage) == 64);
static_assert(sizeof(IR::DescriptorSource::BindlessSampler) == 4);
static_assert(sizeof(IR::ResourceBlock) == 64);
static_assert(sizeof(IR::SrtRead) == 24);
static_assert(sizeof(IR::UniformFillPlan) == 96);
static_assert(sizeof(IR::BindingLayout) == 64);
static_assert(sizeof(IR::DescriptorBinding) == 32);
static_assert(sizeof(IR::CompiledShaderInfo) == 320);
static_assert(sizeof(IR::Inst) == 104);
#endif

template <typename... Args>
void DiskCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

class Writer {
public:
	template <typename T>
	void Pod(const T& value) {
		static_assert(std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>);
		const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
		m_bytes.insert(m_bytes.end(), bytes, bytes + sizeof(T));
	}
	void U32(uint32_t value) { Pod(value); }
	void Bool(bool value) { Pod(static_cast<uint8_t>(value ? 1u : 0u)); }
	template <typename T>
	void PodVector(const std::vector<T>& values) {
		static_assert(std::is_trivially_copyable_v<T> && !std::is_same_v<T, IR::Value>);
		U32(static_cast<uint32_t>(values.size()));
		const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
		m_bytes.insert(m_bytes.end(), bytes, bytes + values.size() * sizeof(T));
	}
	void String(const std::string& value) {
		U32(static_cast<uint32_t>(value.size()));
		m_bytes.insert(m_bytes.end(), value.begin(), value.end());
	}
	void Bytes(std::span<const uint8_t> bytes) {
		m_bytes.insert(m_bytes.end(), bytes.begin(), bytes.end());
	}

	[[nodiscard]] std::vector<uint8_t>& Data() { return m_bytes; }

private:
	std::vector<uint8_t> m_bytes;
};

class Reader {
public:
	explicit Reader(std::span<const uint8_t> data): m_data(data) {}

	template <typename T>
	[[nodiscard]] bool Pod(T& value) {
		static_assert(std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>);
		if (m_data.size() - m_offset < sizeof(T)) {
			return false;
		}
		std::memcpy(&value, m_data.data() + m_offset, sizeof(T));
		m_offset += sizeof(T);
		return true;
	}
	[[nodiscard]] bool U32(uint32_t& value) { return Pod(value); }
	[[nodiscard]] bool Bool(bool& value) {
		uint8_t byte = 0;
		if (!Pod(byte) || byte > 1u) {
			return false;
		}
		value = byte != 0;
		return true;
	}
	template <typename T>
	[[nodiscard]] bool PodVector(std::vector<T>& values) {
		static_assert(std::is_trivially_copyable_v<T> && !std::is_same_v<T, IR::Value>);
		uint32_t count = 0;
		if (!U32(count) || (m_data.size() - m_offset) / sizeof(T) < count) {
			return false;
		}
		values.resize(count);
		std::memcpy(values.data(), m_data.data() + m_offset, count * sizeof(T));
		m_offset += count * sizeof(T);
		return true;
	}
	[[nodiscard]] bool String(std::string& value) {
		uint32_t size = 0;
		if (!U32(size) || m_data.size() - m_offset < size) {
			return false;
		}
		value.assign(reinterpret_cast<const char*>(m_data.data() + m_offset), size);
		m_offset += size;
		return true;
	}
	[[nodiscard]] bool Bytes(std::span<const uint8_t>& bytes, uint32_t size) {
		if (m_data.size() - m_offset < size) {
			return false;
		}
		bytes = m_data.subspan(m_offset, size);
		m_offset += size;
		return true;
	}
	// A count of elements that take at least `min_bytes` each; bounds allocations on bad input.
	[[nodiscard]] bool Count(uint32_t& count, size_t min_bytes) {
		return U32(count) && (m_data.size() - m_offset) / min_bytes >= count;
	}
	[[nodiscard]] bool Done() const { return m_offset == m_data.size(); }

private:
	std::span<const uint8_t> m_data;
	size_t                   m_offset = 0;
};

using InstIndex = std::unordered_map<const IR::Inst*, uint32_t>;

bool WriteValue(Writer& w, IR::Value value, const InstIndex& index) {
	if (value.IsEmpty()) {
		w.U32(static_cast<uint32_t>(IR::Type::Void));
		return true;
	}
	if (const auto* inst = value.TryInstruction(); inst != nullptr) {
		const auto found = index.find(inst);
		if (found == index.end()) {
			return false;
		}
		w.U32(static_cast<uint32_t>(IR::Type::Opaque));
		w.U32(found->second);
		return true;
	}
	const auto type = value.GetType();
	w.U32(static_cast<uint32_t>(type));
	switch (type) {
		case IR::Type::ScalarReg: w.U32(IR::RegIndex(value.ScalarRegister())); return true;
		case IR::Type::VectorReg: w.U32(IR::RegIndex(value.VectorRegister())); return true;
		case IR::Type::U1: w.Bool(value.U1()); return true;
		case IR::Type::U8: w.Pod(value.U8()); return true;
		case IR::Type::U16: w.Pod(value.U16()); return true;
		case IR::Type::F16: w.Pod(value.F16Bits()); return true;
		case IR::Type::U32: w.Pod(value.U32()); return true;
		case IR::Type::F32: w.Pod(std::bit_cast<uint32_t>(value.F32Value())); return true;
		case IR::Type::U64: w.Pod(value.U64()); return true;
		default: return false;
	}
}

bool ReadValue(Reader& r, IR::Value& value, const std::vector<IR::Inst*>& insts) {
	uint32_t type = 0;
	if (!r.U32(type)) {
		return false;
	}
	switch (static_cast<IR::Type>(type)) {
		case IR::Type::Void: value = {}; return true;
		case IR::Type::Opaque: {
			uint32_t index = 0;
			if (!r.U32(index) || index >= insts.size()) {
				return false;
			}
			value = IR::Value(insts[index]);
			return true;
		}
		case IR::Type::ScalarReg: {
			uint32_t reg = 0;
			if (!r.U32(reg) || reg > UINT16_MAX) {
				return false;
			}
			value = IR::Value(static_cast<IR::ScalarReg>(reg));
			return true;
		}
		case IR::Type::VectorReg: {
			uint32_t reg = 0;
			if (!r.U32(reg) || reg > UINT16_MAX) {
				return false;
			}
			value = IR::Value(static_cast<IR::VectorReg>(reg));
			return true;
		}
		case IR::Type::U1: {
			bool bit = false;
			if (!r.Bool(bit)) {
				return false;
			}
			value = IR::Value(bit);
			return true;
		}
		case IR::Type::U8: {
			uint8_t bits = 0;
			if (!r.Pod(bits)) {
				return false;
			}
			value = IR::Value(bits);
			return true;
		}
		case IR::Type::U16:
		case IR::Type::F16: {
			uint16_t bits = 0;
			if (!r.Pod(bits)) {
				return false;
			}
			value = static_cast<IR::Type>(type) == IR::Type::U16 ? IR::Value(bits)
			                                                     : IR::Value::F16(bits);
			return true;
		}
		case IR::Type::U32:
		case IR::Type::F32: {
			uint32_t bits = 0;
			if (!r.Pod(bits)) {
				return false;
			}
			value = static_cast<IR::Type>(type) == IR::Type::U32
			            ? IR::Value(bits)
			            : IR::Value::F32(std::bit_cast<float>(bits));
			return true;
		}
		case IR::Type::U64: {
			uint64_t bits = 0;
			if (!r.Pod(bits)) {
				return false;
			}
			value = IR::Value(bits);
			return true;
		}
		default: return false;
	}
}

void WriteShaderInfo(Writer& w, const IR::ShaderInfo& info) {
	w.PodVector(info.buffers);
	w.U32(static_cast<uint32_t>(info.images.size()));
	for (const auto& image: info.images) {
		w.U32(image.source);
		w.U32(image.first_use_pc);
		w.Pod(image.resource_class);
		w.Pod(image.numeric_class);
		w.Pod(image.dimension);
		w.Pod(image.mip_mode);
		w.U32(image.mip_count);
		w.Pod(image.conversion_format);
		w.U32(image.shader_swizzle);
		w.Bool(image.read);
		w.Bool(image.written);
		w.Bool(image.atomic);
		w.Bool(image.depth_compare);
		w.Bool(image.cube);
		w.Bool(image.r128);
		w.U32(image.indirect_root);
		w.U32(image.indirect_mapping_offset);
		w.U32(image.indirect_search_iterations);
		w.Bool(image.bindless);
		w.PodVector(image.indirect_resources);
	}
	w.PodVector(info.samplers);
	w.PodVector(info.sampled_pairs);
	w.U32(static_cast<uint32_t>(info.inputs.size()));
	for (const auto& input: info.inputs) {
		w.Pod(input.kind);
		w.U32(input.location);
		w.U32(input.component_count);
		w.String(input.debug_name);
		w.Bool(input.per_vertex);
	}
	w.U32(static_cast<uint32_t>(info.outputs.size()));
	for (const auto& output: info.outputs) {
		w.Pod(output.kind);
		w.U32(output.index);
		w.U32(output.location);
		w.String(output.debug_name);
	}
	w.PodVector(info.dma_base_registers);
	w.Pod(info.vertex_fetch_components);
	w.Pod(info.vertex_offset_sgpr);
	w.Pod(info.instance_offset_sgpr);
	w.Bool(info.has_bitwise_xor);
	w.Bool(info.uses_dma);
	w.Bool(info.watchdog_reports);
}

bool ReadShaderInfo(Reader& r, IR::ShaderInfo& info) {
	uint32_t count = 0;
	if (!r.PodVector(info.buffers) || !r.Count(count, 32)) {
		return false;
	}
	info.images.resize(count);
	for (auto& image: info.images) {
		if (!r.U32(image.source) || !r.U32(image.first_use_pc) || !r.Pod(image.resource_class) ||
		    !r.Pod(image.numeric_class) || !r.Pod(image.dimension) || !r.Pod(image.mip_mode) ||
		    !r.U32(image.mip_count) || !r.Pod(image.conversion_format) ||
		    !r.U32(image.shader_swizzle) || !r.Bool(image.read) || !r.Bool(image.written) ||
		    !r.Bool(image.atomic) || !r.Bool(image.depth_compare) || !r.Bool(image.cube) ||
		    !r.Bool(image.r128) || !r.U32(image.indirect_root) ||
		    !r.U32(image.indirect_mapping_offset) || !r.U32(image.indirect_search_iterations) ||
		    !r.Bool(image.bindless) || !r.PodVector(image.indirect_resources)) {
			return false;
		}
	}
	if (!r.PodVector(info.samplers) || !r.PodVector(info.sampled_pairs) || !r.Count(count, 8)) {
		return false;
	}
	info.inputs.resize(count);
	for (auto& input: info.inputs) {
		if (!r.Pod(input.kind) || !r.U32(input.location) || !r.U32(input.component_count) ||
		    !r.String(input.debug_name) || !r.Bool(input.per_vertex)) {
			return false;
		}
	}
	if (!r.Count(count, 8)) {
		return false;
	}
	info.outputs.resize(count);
	for (auto& output: info.outputs) {
		if (!r.Pod(output.kind) || !r.U32(output.index) || !r.U32(output.location) ||
		    !r.String(output.debug_name)) {
			return false;
		}
	}
	return r.PodVector(info.dma_base_registers) && r.Pod(info.vertex_fetch_components) &&
	       r.Pod(info.vertex_offset_sgpr) && r.Pod(info.instance_offset_sgpr) &&
	       r.Bool(info.has_bitwise_xor) && r.Bool(info.uses_dma) && r.Bool(info.watchdog_reports);
}

bool WritePlan(Writer& w, const IR::ResourcePlan& plan) {
	InstIndex index;
	uint32_t  count = 0;
	for (const auto& inst: plan.value_storage) {
		index.emplace(&inst, count++);
	}
	w.Pod(plan.stage);
	w.Pod(plan.shader_hash);
	w.U32(plan.user_data_base);
	w.U32(plan.user_data_count);
	w.U32(count);
	for (const auto& inst: plan.value_storage) {
		w.Pod(inst.GetOpcode());
		w.Pod(inst.Flags<uint64_t>());
	}
	for (const auto& inst: plan.value_storage) {
		// Plan values are detached from blocks; phi operands carry no predecessor.
		if (inst.Parent() != nullptr) {
			return false;
		}
		const bool phi = inst.GetOpcode() == IR::ValueOpcode::Phi;
		if (!phi && inst.NumPhiBlocks() != 0) {
			return false;
		}
		if (phi) {
			if (inst.NumPhiBlocks() != inst.NumArgs()) {
				return false;
			}
			for (size_t i = 0; i < inst.NumPhiBlocks(); i++) {
				if (inst.PhiBlock(i) != nullptr) {
					return false;
				}
			}
		}
		w.U32(static_cast<uint32_t>(inst.NumArgs()));
		for (size_t i = 0; i < inst.NumArgs(); i++) {
			if (!WriteValue(w, inst.Arg(i), index)) {
				return false;
			}
		}
	}
	w.PodVector(plan.memory_info);
	w.U32(static_cast<uint32_t>(plan.descriptor_sources.size()));
	for (const auto& source: plan.descriptor_sources) {
		w.U32(source.dword_count);
		for (const auto& dword: source.dwords) {
			if (!WriteValue(w, dword, index)) {
				return false;
			}
		}
		w.Bool(source.indirect_image.has_value());
		if (source.indirect_image.has_value()) {
			const auto& image = *source.indirect_image;
			w.U32(image.material_source);
			w.U32(image.table_source);
			w.U32(image.selector_stride);
			w.U32(image.selector_offset);
			w.U32(image.table_offset);
			w.U32(image.key_shift);
			w.U32(image.key_mask);
			w.Bool(image.bindless);
			if (!WriteValue(w, image.key_count, index) ||
			    !WriteValue(w, image.selector_mask, index)) {
				return false;
			}
		}
		w.Bool(source.bindless_sampler.has_value());
		if (source.bindless_sampler.has_value()) {
			w.U32(source.bindless_sampler->table_offset);
		}
	}
	w.U32(static_cast<uint32_t>(plan.control_flow.size()));
	for (const auto& block: plan.control_flow) {
		if (!WriteValue(w, block.condition, index)) {
			return false;
		}
		w.PodVector(block.successors);
		w.PodVector(block.sources);
	}
	w.U32(static_cast<uint32_t>(plan.srt_reads.size()));
	for (const auto& read: plan.srt_reads) {
		if (!WriteValue(w, read.value, index)) {
			return false;
		}
		w.U32(read.flat_offset);
	}
	w.PodVector(plan.clean_flat_slots);
	w.Bool(plan.requires_specialization_memory);
	w.Bool(plan.capture_specialization_reads);
	w.Bool(plan.descriptor_phi_under_writes);
	w.Bool(plan.has_uniform_buffer_reads);
	w.Bool(plan.bindless_images);
	w.Bool(plan.srt_plan_complete);
	w.Bool(plan.resource_tracking_complete);
	WriteShaderInfo(w, plan.info);
	w.Pod(plan.uniform_fill.fill);
	for (const auto& value: plan.uniform_fill.values) {
		if (!WriteValue(w, value, index)) {
			return false;
		}
	}
	return true;
}

bool ReadPlan(Reader& r, IR::ResourcePlan& plan) {
	uint32_t count = 0;
	if (!r.Pod(plan.stage) || !r.Pod(plan.shader_hash) || !r.U32(plan.user_data_base) ||
	    !r.U32(plan.user_data_count) || !r.Count(count, sizeof(uint32_t) + sizeof(uint64_t))) {
		return false;
	}
	std::vector<IR::Inst*> insts;
	insts.reserve(count);
	for (uint32_t i = 0; i < count; i++) {
		IR::ValueOpcode opcode {};
		uint64_t        flags = 0;
		if (!r.Pod(opcode) || !r.Pod(flags) ||
		    static_cast<uint32_t>(opcode) >= static_cast<uint32_t>(IR::ValueOpcode::Count)) {
			return false;
		}
		insts.push_back(&plan.value_storage.emplace_back(opcode, flags));
	}
	for (auto* inst: insts) {
		uint32_t args = 0;
		if (!r.Count(args, sizeof(uint32_t))) {
			return false;
		}
		const bool phi   = inst->GetOpcode() == IR::ValueOpcode::Phi;
		const auto fixed = IR::NumArgsOf(inst->GetOpcode());
		if (!phi && fixed != std::numeric_limits<size_t>::max() && fixed != args) {
			return false;
		}
		for (uint32_t i = 0; i < args; i++) {
			IR::Value value;
			if (!ReadValue(r, value, insts)) {
				return false;
			}
			if (phi) {
				inst->AddPhiOperand(nullptr, value);
			} else {
				inst->SetArg(i, value);
			}
		}
	}
	if (!r.PodVector(plan.memory_info) || !r.Count(count, 8)) {
		return false;
	}
	plan.descriptor_sources.resize(count);
	for (auto& source: plan.descriptor_sources) {
		bool has = false;
		if (!r.U32(source.dword_count) || source.dword_count > source.dwords.size()) {
			return false;
		}
		for (auto& dword: source.dwords) {
			if (!ReadValue(r, dword, insts)) {
				return false;
			}
		}
		if (!r.Bool(has)) {
			return false;
		}
		if (has) {
			auto& image = source.indirect_image.emplace();
			if (!r.U32(image.material_source) || !r.U32(image.table_source) ||
			    !r.U32(image.selector_stride) || !r.U32(image.selector_offset) ||
			    !r.U32(image.table_offset) || !r.U32(image.key_shift) || !r.U32(image.key_mask) ||
			    !r.Bool(image.bindless) || !ReadValue(r, image.key_count, insts) ||
			    !ReadValue(r, image.selector_mask, insts)) {
				return false;
			}
		}
		if (!r.Bool(has)) {
			return false;
		}
		if (has && !r.U32(source.bindless_sampler.emplace().table_offset)) {
			return false;
		}
	}
	if (!r.Count(count, 12)) {
		return false;
	}
	plan.control_flow.resize(count);
	for (auto& block: plan.control_flow) {
		if (!ReadValue(r, block.condition, insts) || !r.PodVector(block.successors) ||
		    !r.PodVector(block.sources)) {
			return false;
		}
	}
	if (!r.Count(count, 8)) {
		return false;
	}
	plan.srt_reads.resize(count);
	for (auto& read: plan.srt_reads) {
		if (!ReadValue(r, read.value, insts) || !r.U32(read.flat_offset)) {
			return false;
		}
	}
	if (!r.PodVector(plan.clean_flat_slots) || !r.Bool(plan.requires_specialization_memory) ||
	    !r.Bool(plan.capture_specialization_reads) || !r.Bool(plan.descriptor_phi_under_writes) ||
	    !r.Bool(plan.has_uniform_buffer_reads) || !r.Bool(plan.bindless_images) ||
	    !r.Bool(plan.srt_plan_complete) || !r.Bool(plan.resource_tracking_complete) ||
	    !ReadShaderInfo(r, plan.info) || !r.Pod(plan.uniform_fill.fill)) {
		return false;
	}
	for (auto& value: plan.uniform_fill.values) {
		if (!ReadValue(r, value, insts)) {
			return false;
		}
	}
	return true;
}

void WriteCompiledInfo(Writer& w, const IR::CompiledShaderInfo& program) {
	w.Pod(program.stage);
	w.Pod(program.shader_hash);
	w.U32(program.wave_size);
	w.U32(program.user_data_base);
	w.U32(program.user_data_count);
	w.U32(program.scratch_dwords);
	w.U32(program.param_export_mask);
	WriteShaderInfo(w, program.info);
	const auto& bindings = program.bindings;
	w.U32(bindings.push_data_start_dword);
	w.U32(bindings.memory_offset_dword);
	w.U32(bindings.memory_offset_count);
	w.Bool(bindings.has_dispatch_dimensions);
	w.Bool(bindings.has_write_reports);
	w.PodVector(bindings.user_data_registers);
	w.U32(static_cast<uint32_t>(bindings.descriptors.size()));
	for (const auto& descriptor: bindings.descriptors) {
		w.Pod(descriptor.kind);
		w.PodVector(descriptor.resources);
	}
}

bool ReadCompiledInfo(Reader& r, IR::CompiledShaderInfo& program) {
	auto&    bindings = program.bindings;
	uint32_t count    = 0;
	if (!r.Pod(program.stage) || !r.Pod(program.shader_hash) || !r.U32(program.wave_size) ||
	    !r.U32(program.user_data_base) || !r.U32(program.user_data_count) ||
	    !r.U32(program.scratch_dwords) || !r.U32(program.param_export_mask) ||
	    !ReadShaderInfo(r, program.info) || !r.U32(bindings.push_data_start_dword) ||
	    !r.U32(bindings.memory_offset_dword) || !r.U32(bindings.memory_offset_count) ||
	    !r.Bool(bindings.has_dispatch_dimensions) || !r.Bool(bindings.has_write_reports) ||
	    !r.PodVector(bindings.user_data_registers) || !r.Count(count, 8)) {
		return false;
	}
	bindings.descriptors.resize(count);
	for (auto& descriptor: bindings.descriptors) {
		if (!r.Pod(descriptor.kind) || !r.PodVector(descriptor.resources)) {
			return false;
		}
	}
	return true;
}

// The emulator executable's contents. Any rebuild (a translation change included, committed or
// not) gives a new identity and starts the file over.
std::optional<std::array<uint64_t, 2>> BuildId() {
#ifdef _WIN32
	std::wstring path(32768, L'\0');
	const auto   length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
	if (length == 0 || length >= path.size()) {
		return std::nullopt;
	}
	path.resize(length);
	const std::filesystem::path exe(path);
#else
	std::error_code             error;
	const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", error);
	if (error) {
		return std::nullopt;
	}
#endif
	Common::File file(exe, Common::File::Mode::Read);
	if (file.IsInvalid()) {
		return std::nullopt;
	}
	auto* state = XXH3_createState();
	if (state == nullptr || XXH3_128bits_reset(state) != XXH_OK) {
		XXH3_freeState(state);
		return std::nullopt;
	}
	std::vector<uint8_t> chunk(4u << 20u);
	uint64_t             total = 0;
	for (;;) {
		uint32_t read = 0;
		file.Read(chunk.data(), static_cast<uint32_t>(chunk.size()), &read);
		if (read == 0) {
			break;
		}
		XXH3_128bits_update(state, chunk.data(), read);
		total += read;
	}
	const auto size = file.Size();
	file.Close();
	const auto hash = XXH3_128bits_digest(state);
	XXH3_freeState(state);
	if (total != size || total == 0) {
		return std::nullopt;
	}
	return std::array<uint64_t, 2> {hash.low64, hash.high64};
}

} // namespace

ShaderDiskCache::Key ShaderDiskCache::KeyBuilder::Finish() const {
	const auto hash = XXH3_128bits(m_bytes.data(), m_bytes.size());
	return {.low = hash.low64, .high = hash.high64};
}

uint64_t ShaderDiskCache::EnvironmentHash() {
	static const uint64_t hash = [] {
		std::vector<std::string> variables;
		// Read at use by the recompiler; listed so they count even without an environment block.
		for (const char* name: {"KYTY_WRITE_REPORTS", "KYTY_TRANSLATE_BVH", "KYTY_BINDLESS_SAMPLERS",
		                        "KYTY_MESH_CULL", "KYTY_IGNORE_WRITTEN_OVERLAP"}) {
			if (const char* value = std::getenv(name); value != nullptr) {
				variables.push_back(fmt::format("{}={}", name, value));
			}
		}
		// Every other KYTY_ switch too: one that does not affect translation only costs a miss.
#ifdef _WIN32
		if (auto* block = GetEnvironmentStringsA(); block != nullptr) {
			for (const char* entry = block; *entry != '\0'; entry += std::strlen(entry) + 1) {
				if (std::string_view(entry).starts_with("KYTY_")) {
					variables.emplace_back(entry);
				}
			}
			FreeEnvironmentStringsA(block);
		}
#else
		for (auto** entry = environ; entry != nullptr && *entry != nullptr; entry++) {
			if (std::string_view(*entry).starts_with("KYTY_")) {
				variables.emplace_back(*entry);
			}
		}
#endif
		std::ranges::sort(variables);
		std::string text = fmt::format("format={}\n", FormatVersion);
		for (const auto& variable: variables) {
			text += variable;
			text += '\n';
		}
		return XXH3_64bits(text.data(), text.size());
	}();
	return hash;
}

std::unique_ptr<ShaderDiskCache> ShaderDiskCache::Open(const std::filesystem::path& path) {
	if (const char* value = std::getenv("KYTY_NO_SHADER_DISK_CACHE");
	    value != nullptr && std::string_view(value) != "0") {
		DiskCacheLog("Shader disk cache: disabled (KYTY_NO_SHADER_DISK_CACHE)");
		return nullptr;
	}
	const auto build_id = BuildId();
	if (!build_id.has_value()) {
		DiskCacheLog("Shader disk cache: disabled (cannot identify the emulator build)");
		return nullptr;
	}
	auto cache        = std::make_unique<ShaderDiskCache>();
	cache->m_path     = path;
	cache->m_build_id = *build_id;
	if (!cache->Load()) {
		return nullptr;
	}
	return cache;
}

ShaderDiskCache::~ShaderDiskCache() {
	Write();
	DiskCacheLog("Shader disk cache: {} hits, {} new records this session", m_hits, m_stored);
}

bool ShaderDiskCache::Load() {
	const auto path_text = Common::PathToString(m_path);
	if (!Common::File::IsFileExisting(m_path)) {
		DiskCacheLog("Shader disk cache: initializing {}", path_text);
		return true;
	}
	Common::File file(m_path, Common::File::Mode::Read);
	if (file.IsInvalid()) {
		DiskCacheLog("Shader disk cache: disabled (cannot read {})", path_text);
		return false;
	}
	const auto size = file.Size();
	if (size < sizeof(FileHeader) || size > MaxFileSize) {
		file.Close();
		DiskCacheLog("Shader disk cache: starting {} over (invalid size)", path_text);
		return true;
	}
	std::vector<uint8_t> data(size);
	uint32_t             read = 0;
	file.Read(data.data(), static_cast<uint32_t>(size), &read);
	file.Close();
	FileHeader header;
	if (read != size) {
		DiskCacheLog("Shader disk cache: starting {} over (read failed)", path_text);
		return true;
	}
	std::memcpy(&header, data.data(), sizeof(header));
	if (header.magic != FileMagic || header.version != FormatVersion ||
	    header.header_size != sizeof(FileHeader) || header.build_low != m_build_id[0] ||
	    header.build_high != m_build_id[1]) {
		DiskCacheLog("Shader disk cache: starting {} over (different emulator build or format)",
		             path_text);
		return true;
	}
	uint64_t offset       = sizeof(FileHeader);
	uint32_t plans        = 0;
	uint32_t permutations = 0;
	while (size - offset >= sizeof(RecordHeader)) {
		RecordHeader record;
		std::memcpy(&record, data.data() + offset, sizeof(record));
		if (record.magic != RecordMagic || record.reserved != 0 || record.size > MaxPayloadSize ||
		    size - offset - sizeof(RecordHeader) < record.size ||
		    (record.kind != RecordPlan && record.kind != RecordPermutation)) {
			break;
		}
		const auto* payload = data.data() + offset + sizeof(RecordHeader);
		if (XXH3_64bits(payload, record.size) != record.hash) {
			break;
		}
		const Key key {.low = record.key_low, .high = record.key_high};
		if (record.kind == RecordPlan) {
			m_plans[key].assign(payload, payload + record.size);
			plans++;
		} else {
			m_permutations[key].emplace_back(payload, payload + record.size);
			permutations++;
		}
		offset += sizeof(RecordHeader) + record.size;
	}
	m_file_end = offset;
	DiskCacheLog("Shader disk cache: loaded {} programs and {} permutations from {}{}", plans,
	             permutations, path_text,
	             offset != size ? fmt::format(" (ignored {} trailing bytes)", size - offset) : "");
	return true;
}

std::optional<IR::ResourcePlan> ShaderDiskCache::FindPlan(const Key& key) {
	const auto found = m_plans.find(key);
	if (found == m_plans.end()) {
		return std::nullopt;
	}
	IR::ResourcePlan plan;
	Reader           reader(found->second);
	if (!ReadPlan(reader, plan) || !reader.Done()) {
		DiskCacheLog("Shader disk cache: dropping an unreadable program record");
		m_plans.erase(found);
		return std::nullopt;
	}
	m_hits++;
	return plan;
}

std::vector<uint8_t> ShaderDiskCache::SerializeSpecialization(
    const IR::ResourceSpecialization& spec) {
	Writer w;
	w.PodVector(spec.buffers);
	w.PodVector(spec.images);
	w.PodVector(spec.samplers);
	return std::move(w.Data());
}

std::optional<ShaderDiskCache::Permutation> ShaderDiskCache::DecodePermutation(
    const std::vector<uint8_t>& record, const std::vector<uint8_t>& spec_bytes) const {
	Reader                   reader(record);
	uint32_t                 spec_size = 0;
	std::span<const uint8_t> stored_spec;
	if (!reader.U32(spec_size) || spec_size != spec_bytes.size() ||
	    !reader.Bytes(stored_spec, spec_size) ||
	    !std::ranges::equal(stored_spec, std::span<const uint8_t>(spec_bytes))) {
		return std::nullopt;
	}
	Permutation permutation;
	if (!ReadCompiledInfo(reader, permutation.program) || !reader.PodVector(permutation.spirv) ||
	    !reader.Done() || permutation.spirv.size() < 5 || permutation.spirv[0] != SpirvMagic) {
		return std::nullopt;
	}
	return permutation;
}

void ShaderDiskCache::StorePlan(const Key& key, const IR::ResourcePlan& plan) {
	if (m_plans.contains(key)) {
		return;
	}
	Writer writer;
	if (!WritePlan(writer, plan)) {
		return;
	}
	// A plan is stored only when reading it back reproduces it exactly.
	IR::ResourcePlan check;
	Reader           reader(writer.Data());
	Writer           rewritten;
	if (!ReadPlan(reader, check) || !reader.Done() || !WritePlan(rewritten, check) ||
	    rewritten.Data() != writer.Data() || !(check.info == plan.info) ||
	    !(check.memory_info == plan.memory_info)) {
		static bool logged = false;
		if (!std::exchange(logged, true)) {
			DiskCacheLog("Shader disk cache: a resource plan does not round-trip; not stored");
		}
		return;
	}
	if (writer.Data().size() > MaxPayloadSize) {
		return;
	}
	Append(RecordPlan, key, writer.Data());
	m_plans.emplace(key, std::move(writer.Data()));
}

void ShaderDiskCache::StorePermutation(const Key& key, const IR::ResourceSpecialization& spec,
                                       const IR::CompiledShaderInfo& program,
                                       std::span<const uint32_t>     spirv) {
	const auto spec_bytes = SerializeSpecialization(spec);
	Writer     writer;
	writer.U32(static_cast<uint32_t>(spec_bytes.size()));
	writer.Bytes(spec_bytes);
	WriteCompiledInfo(writer, program);
	writer.PodVector(std::vector<uint32_t>(spirv.begin(), spirv.end()));
	if (writer.Data().size() > MaxPayloadSize) {
		return;
	}
	const auto check = DecodePermutation(writer.Data(), spec_bytes);
	Writer     rewritten;
	if (check.has_value()) {
		WriteCompiledInfo(rewritten, check->program);
	}
	Writer original;
	WriteCompiledInfo(original, program);
	if (!check.has_value() || rewritten.Data() != original.Data() ||
	    !(check->program.info == program.info) || !(check->program.bindings == program.bindings)) {
		static bool logged = false;
		if (!std::exchange(logged, true)) {
			DiskCacheLog("Shader disk cache: a compiled shader does not round-trip; not stored");
		}
		return;
	}
	Append(RecordPermutation, key, writer.Data());
	m_permutations[key].push_back(std::move(writer.Data()));
}

void ShaderDiskCache::Append(uint32_t kind, const Key& key, std::span<const uint8_t> payload) {
	RecordHeader record;
	record.kind     = kind;
	record.key_low  = key.low;
	record.key_high = key.high;
	record.size     = static_cast<uint32_t>(payload.size());
	record.hash     = XXH3_64bits(payload.data(), payload.size());
	const auto* bytes = reinterpret_cast<const uint8_t*>(&record);
	m_pending.insert(m_pending.end(), bytes, bytes + sizeof(record));
	m_pending.insert(m_pending.end(), payload.begin(), payload.end());
	m_stored++;
}

void ShaderDiskCache::WriteIfDue() {
	if (m_pending.empty()) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - m_written < std::chrono::seconds(5)) {
		return;
	}
	m_written = now;
	Write();
}

void ShaderDiskCache::Write() {
	if (m_pending.empty() || m_write_failed) {
		return;
	}
	if (m_file_end + m_pending.size() > MaxFileSize) {
		m_pending.clear();
		return;
	}
	Common::File file;
	bool         ok = true;
	if (m_file_end != 0 && !Common::File::IsFileExisting(m_path)) {
		// Deleted while running: start over with what is pending (earlier records are lost).
		m_file_end = 0;
	}
	if (m_file_end == 0) {
		// Start the file over: a new file, another build's, or an unreadable one.
		ok = Common::File::CreateDirectories(m_path.parent_path()) && file.Create(m_path);
		if (ok) {
			FileHeader header;
			header.build_low  = m_build_id[0];
			header.build_high = m_build_id[1];
			uint32_t written  = 0;
			file.Write(&header, sizeof(header), &written);
			ok         = written == sizeof(header);
			m_file_end = sizeof(header);
		}
	} else {
		ok = file.Open(m_path, Common::File::Mode::ReadWrite) && file.Seek(m_file_end);
	}
	if (ok) {
		uint32_t written = 0;
		file.Write(m_pending.data(), static_cast<uint32_t>(m_pending.size()), &written);
		ok = written == m_pending.size();
	}
	if (ok) {
		m_file_end += m_pending.size();
		// Drop a torn tail from an earlier session past the records just written.
		if (file.Size() > m_file_end) {
			ok = file.Truncate(m_file_end);
		}
		ok = file.Flush() && ok;
	}
	if (!file.IsInvalid()) {
		file.Close();
	}
	m_pending.clear();
	if (!ok) {
		m_write_failed = true;
		DiskCacheLog("Shader disk cache: failed to write {}; keeping new shaders in memory only",
		             Common::PathToString(m_path));
	}
}

} // namespace Libs::Graphics
