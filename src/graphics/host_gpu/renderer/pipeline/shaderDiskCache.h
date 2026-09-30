#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERDISKCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERDISKCACHE_H_

#include "common/common.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// Translated shaders kept across runs (_PipelineCache\<title>.spv.bin): per program, the resource
// plan the recompiler extracts, and per permutation the compiled shader info and SPIR-V. A hit
// skips the translation and the SPIR-V emission of a shader the first time a session uses it.
//
// Correctness rests on the key: the file header carries a hash of the emulator executable (any
// rebuild starts the file over), and each record's key hashes the guest code and every static
// input of the translation (see ProgramCache). Records are appended; a truncated or corrupt tail
// is ignored. Only the GPU thread uses it (with the program cache). KYTY_NO_SHADER_DISK_CACHE=1
// turns it off.
class ShaderDiskCache {
public:
	struct Key {
		uint64_t low  = 0;
		uint64_t high = 0;

		bool operator==(const Key&) const = default;
	};

	// Collects the bytes a key is the hash of (XXH3-128).
	class KeyBuilder {
	public:
		template <typename T>
		void Add(const T& value) {
			static_assert(std::is_trivially_copyable_v<T>);
			const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
			m_bytes.insert(m_bytes.end(), bytes, bytes + sizeof(T));
		}
		void AddWords(std::span<const uint32_t> words) {
			Add(static_cast<uint64_t>(words.size()));
			const auto* bytes = reinterpret_cast<const uint8_t*>(words.data());
			m_bytes.insert(m_bytes.end(), bytes, bytes + words.size_bytes());
		}
		[[nodiscard]] Key Finish() const;

	private:
		std::vector<uint8_t> m_bytes;
	};

	// The KYTY_* environment (which switches translation behaviour) and the format version.
	[[nodiscard]] static uint64_t EnvironmentHash();

	struct Permutation {
		ShaderRecompiler::IR::CompiledShaderInfo program;
		std::vector<uint32_t>                    spirv;
	};

	// Returns null when the cache is disabled (switch, no title, or an unusable file location).
	static std::unique_ptr<ShaderDiskCache> Open(const std::filesystem::path& path);

	ShaderDiskCache() = default;
	~ShaderDiskCache();
	KYTY_CLASS_NO_COPY(ShaderDiskCache);

	[[nodiscard]] std::optional<ShaderRecompiler::IR::ResourcePlan> FindPlan(const Key& key);
	// A permutation stored for this program with the same specialization whose push data start
	// satisfies `matches` (the predicate the in-memory cache applies to its own permutations).
	template <typename Match>
	[[nodiscard]] std::optional<Permutation>
	FindPermutation(const Key& key, const ShaderRecompiler::IR::ResourceSpecialization& spec,
	                Match&& matches) {
		const auto found = m_permutations.find(key);
		if (found == m_permutations.end()) {
			return std::nullopt;
		}
		const auto spec_bytes = SerializeSpecialization(spec);
		for (const auto& record: found->second) {
			auto permutation = DecodePermutation(record, spec_bytes);
			if (permutation.has_value() && matches(permutation->program)) {
				m_hits++;
				return permutation;
			}
		}
		return std::nullopt;
	}

	void StorePlan(const Key& key, const ShaderRecompiler::IR::ResourcePlan& plan);
	void StorePermutation(const Key& key, const ShaderRecompiler::IR::ResourceSpecialization& spec,
	                      const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                      std::span<const uint32_t>                       spirv);

	// Appends the records added since the last write: at most every few seconds while shaders
	// are being compiled, and always at shutdown.
	void WriteIfDue();
	void Write();

private:
	struct KeyHash {
		std::size_t operator()(const Key& key) const {
			return static_cast<std::size_t>(key.low ^ (key.high * 0x9e3779b97f4a7c15ull));
		}
	};

	static std::vector<uint8_t> SerializeSpecialization(
	    const ShaderRecompiler::IR::ResourceSpecialization& spec);
	[[nodiscard]] std::optional<Permutation> DecodePermutation(
	    const std::vector<uint8_t>& record, const std::vector<uint8_t>& spec_bytes) const;
	void Append(uint32_t kind, const Key& key, std::span<const uint8_t> payload);
	bool Load();

	std::filesystem::path m_path;
	std::array<uint64_t, 2> m_build_id {};
	std::unordered_map<Key, std::vector<uint8_t>, KeyHash>              m_plans;
	std::unordered_map<Key, std::vector<std::vector<uint8_t>>, KeyHash> m_permutations;
	std::vector<uint8_t>                                                m_pending;
	// Where the next records go: the end of the valid records read at startup (a torn tail is
	// overwritten), or 0 when the file has to be started over.
	uint64_t                                                            m_file_end = 0;
	bool                                                                m_write_failed = false;
	std::chrono::steady_clock::time_point m_written = std::chrono::steady_clock::now();
	uint64_t                              m_hits    = 0;
	uint64_t                              m_stored  = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERDISKCACHE_H_
