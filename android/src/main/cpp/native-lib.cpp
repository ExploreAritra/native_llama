#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <android/log.h>
#include <unistd.h>
#include <sys/stat.h>
#include <chrono>
#include <thread>
#include "llama.h"
#include "ggml-backend.h"

// --- MTMD Headers ---
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"  // common_chat_* — Jinja chat-template formatting (like mtmd-cli)

#define TAG "NATIVE_LLAMA_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Forward llama.cpp / ggml internal diagnostics to logcat. Without this, all of
// llama.cpp's own logging (Vulkan device detection, "offloaded X/Y layers to
// GPU", per-backend buffer sizes, per-token timings) is written to stderr and
// silently dropped by Android. Routed under the LLAMA_CPP tag so it can be
// filtered separately from this bridge's own NATIVE_LLAMA_JNI messages.
static void llama_to_logcat(ggml_log_level level, const char *text, void * /*user*/) {
    int prio = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR
             : level == GGML_LOG_LEVEL_WARN  ? ANDROID_LOG_WARN
             : ANDROID_LOG_INFO;
    __android_log_print(prio, "LLAMA_CPP", "%s", text);
}

static llama_model * model = nullptr;
static llama_context * ctx = nullptr;
static llama_model * model_draft = nullptr;
static llama_context * ctx_draft = nullptr;
static mtmd_context * mtmd_ctx = nullptr;
static bool stop_generation = false;
static int32_t active_n_ctx = 0; // resolved context window of the live ctx (for resetContext)

double getPhysicalMemoryGB() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return (double)pages * (double)page_size / (1024.0 * 1024.0 * 1024.0);
    }
    return 4.0;
}

int getPerformanceCores() {
#ifdef __ANDROID__
    // On Android, we can try to guess based on CPU frequencies or just return a safe number.
    // However, sysconf(_SC_NPROCESSORS_CONF) gives all cores.
    // For many ARM chips (Big.Little), the higher indexed cores are often the performance ones.
    // A better way is to check /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq
    int total_cores = sysconf(_SC_NPROCESSORS_CONF);
    if (total_cores <= 4) return total_cores;
    // Typical octa-core: 4 efficiency + 4 performance.
    return total_cores / 2;
#else
    return sysconf(_SC_NPROCESSORS_CONF);
#endif
}

double getFileSizeGB(const char* path) {
    struct stat stat_buf;
    int rc = stat(path, &stat_buf);
    return rc == 0 ? stat_buf.st_size / (1024.0 * 1024.0 * 1024.0) : 0.0;
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_timebox_native_1llama_NativeLlamaPlugin_initLlama(JNIEnv *env, jobject thiz, jstring model_path, jint n_ctx, jint n_threads, jint n_gpu_layers) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Initializing llama model from: %s", path);

    double memoryGB = getPhysicalMemoryGB();
    double fileSizeGB = getFileSizeGB(path);

    if (fileSizeGB > 0 && fileSizeGB > (memoryGB * 0.65)) {
        LOGE("RAM SHIELD: Model size (%.2f GB) exceeds safe limits for device RAM (%.2f GB). Aborting load.", fileSizeGB, memoryGB);
        env->ReleaseStringUTFChars(model_path, path);
        return JNI_FALSE;
    }

    if (ctx) { llama_free(ctx); ctx = nullptr; }
    if (model) { llama_model_free(model); model = nullptr; }

    // Route llama.cpp + ggml logs to logcat (LLAMA_CPP tag) so the Vulkan device
    // detection and layer-offload lines are visible for GPU-acceleration debugging.
    llama_log_set(llama_to_logcat, nullptr);
    ggml_log_set(llama_to_logcat, nullptr);

    llama_backend_init();
    ggml_backend_load_all();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    model = llama_model_load_from_file(path, mparams);

    // GPU -> CPU fallback (model load). When a GPU offload is requested but the
    // model fails to load, retry once forcing CPU-only. This rescues the case
    // where a Vulkan device IS present but the GPU load fails (out-of-memory,
    // driver/shader error). The other case -- no usable Vulkan device at all --
    // does NOT hit this path: ggml already keeps all layers on the CPU and the
    // load succeeds. mparams.n_gpu_layers is left at 0 so the context below is
    // built for the CPU too.
    if (model == nullptr && n_gpu_layers != 0) {
        LOGE("GPU model load failed; falling back to CPU (n_gpu_layers=0)");
        mparams.n_gpu_layers = 0;
        model = llama_model_load_from_file(path, mparams);
    }

    if (model == nullptr) {
        LOGE("Failed to load model: %s", path);
        env->ReleaseStringUTFChars(model_path, path);
        return JNI_FALSE;
    }

    auto cparams = llama_context_default_params();
    cparams.n_threads = n_threads > 0 ? n_threads : 4;
    cparams.n_batch = 512;
    cparams.embeddings = true;

    if (n_ctx > 0) {
        cparams.n_ctx = n_ctx;
    } else {
        int32_t dynamic_n_ctx = 4096;
        if (memoryGB >= 7.5) { dynamic_n_ctx = 8192; }
        if (memoryGB >= 11.5) { dynamic_n_ctx = 16384; }
        cparams.n_ctx = dynamic_n_ctx;
    }

    ctx = llama_init_from_model(model, cparams);

    // GPU -> CPU fallback (context/KV allocation). If the model loaded on the
    // GPU but the context failed -- typically the KV cache + compute buffers
    // couldn't be allocated on the GPU -- reload the whole model on CPU (its
    // tensors are already on the GPU, so they must be re-placed) and rebuild the
    // context. mparams.n_gpu_layers != 0 means the model above loaded on the GPU.
    if (ctx == nullptr && mparams.n_gpu_layers != 0) {
        LOGE("GPU context creation failed; reloading model on CPU");
        llama_model_free(model);
        mparams.n_gpu_layers = 0;
        model = llama_model_load_from_file(path, mparams);
        if (model != nullptr) {
            ctx = llama_init_from_model(model, cparams);
        }
    }

    if (ctx == nullptr) {
        LOGE("Failed to create context");
        if (model) { llama_model_free(model); }
        model = nullptr;
        env->ReleaseStringUTFChars(model_path, path);
        return JNI_FALSE;
    }

    active_n_ctx = cparams.n_ctx; // remember for a later resetContext
    env->ReleaseStringUTFChars(model_path, path);
    LOGI("Llama model initialized successfully");
    return JNI_TRUE;
}

// Recreate ONLY the llama context, keeping the model weights and the mtmd vision
// projector resident. A fresh context resets the KV cache and M-RoPE position
// state so the next generation (e.g. the next page image) starts clean WITHOUT
// the multi-GB weight + projector reload that dispose + init costs. llama.cpp's
// per-generation memory_clear does not reliably reset M-RoPE positions on a
// reused vision context after a large/aborted decode; a brand-new context does.
JNIEXPORT jboolean JNICALL
Java_com_timebox_native_1llama_NativeLlamaPlugin_resetContext(JNIEnv *env, jobject thiz, jint n_ctx) {
    if (model == nullptr) return JNI_FALSE; // nothing loaded — caller falls back

    if (ctx) { llama_free(ctx); ctx = nullptr; }

    // Rebuild the context params exactly as initLlama did.
    auto cparams = llama_context_default_params();
    cparams.n_threads = 4;
    cparams.n_batch = 512;
    cparams.embeddings = true;
    cparams.n_ctx = n_ctx > 0 ? n_ctx : (active_n_ctx > 0 ? active_n_ctx : 4096);

    ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        LOGE("resetContext: failed to recreate context");
        return JNI_FALSE;
    }
    active_n_ctx = cparams.n_ctx;
    LOGI("resetContext: context recreated (n_ctx=%d)", cparams.n_ctx);
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_timebox_native_1llama_NativeLlamaPlugin_initVision(JNIEnv *env, jobject thiz, jstring mmproj_path) {
    const char *path = env->GetStringUTFChars(mmproj_path, nullptr);
    LOGI("Initializing Vision model from: %s", path);

    if (mtmd_ctx) { mtmd_free(mtmd_ctx); mtmd_ctx = nullptr; }

    if (model == nullptr) {
        LOGE("Error: Main text model must be loaded first.");
        env->ReleaseStringUTFChars(mmproj_path, path);
        return JNI_FALSE;
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    // Vision encoder on the GPU (Vulkan). The A19-Metal "tensor API" bug that made
    // iOS extraction return all-empty fields is Metal-only (fixed there via
    // GGML_METAL_TENSOR_DISABLE); it does not apply to the Vulkan backend, so
    // Android keeps the vision encoder on the GPU for speed.
    mtmd_params.use_gpu = true;
    // Full-page document reads (Qwen2.5-VL) need enough vision tokens to keep
    // fine print legible; 1536 handles a dense A4 page without tiling.
    mtmd_params.image_max_tokens = 1536;
    mtmd_params.image_min_tokens = 256;

    mtmd_ctx = mtmd_init_from_file(path, model, mtmd_params);

    if (mtmd_ctx == nullptr) {
        LOGE("Failed to load MTMD model: %s", path);
        env->ReleaseStringUTFChars(mmproj_path, path);
        return JNI_FALSE;
    }

    env->ReleaseStringUTFChars(mmproj_path, path);
    LOGI("Vision model initialized successfully");
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_timebox_native_1llama_NativeLlamaPlugin_initDraftModel(JNIEnv *env, jobject thiz, jstring model_path, jint n_ctx, jint n_threads, jint n_gpu_layers) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);

    if (ctx_draft) { llama_free(ctx_draft); ctx_draft = nullptr; }
    if (model_draft) { llama_model_free(model_draft); model_draft = nullptr; }

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    model_draft = llama_model_load_from_file(path, mparams);

    if (model_draft == nullptr) {
        env->ReleaseStringUTFChars(model_path, path);
        return JNI_FALSE;
    }

    auto cparams = llama_context_default_params();
    cparams.n_threads = n_threads > 0 ? n_threads : 4;
    cparams.embeddings = true;
    cparams.n_batch = 256;

    if (n_ctx > 0) {
        cparams.n_ctx = n_ctx;
    } else {
        double memoryGB = getPhysicalMemoryGB();
        int32_t dynamic_n_ctx = 4096;
        if (memoryGB >= 7.5) dynamic_n_ctx = 8192;
        if (memoryGB >= 11.5) dynamic_n_ctx = 16384;
        cparams.n_ctx = dynamic_n_ctx;
    }

    ctx_draft = llama_init_from_model(model_draft, cparams);
    if (ctx_draft == nullptr) {
        llama_model_free(model_draft);
        model_draft = nullptr;
        env->ReleaseStringUTFChars(model_path, path);
        return JNI_FALSE;
    }

    env->ReleaseStringUTFChars(model_path, path);
    return JNI_TRUE;
}

JNIEXPORT jdoubleArray JNICALL
Java_com_timebox_native_1llama_NativeLlamaPlugin_getEmbedding(JNIEnv *env, jobject thiz, jstring text) {
    if (ctx == nullptr || model == nullptr) return nullptr;

    const char * prompt = env->GetStringUTFChars(text, nullptr);
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // Get token count
    int n_tokens = -llama_tokenize(vocab, prompt, strlen(prompt), NULL, 0, true, true);
    if (n_tokens < 0) n_tokens = -n_tokens; // Ensure it's positive

    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, true);

    // CRITICAL FIX: Clear the KV cache so embeddings don't stack up indefinitely!
    llama_memory_seq_rm(llama_get_memory(ctx), -1, -1, -1);

    uint32_t n_batch_size = llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(n_batch_size, 0, 1);

    int n_eval = 0;
    while (n_eval < (int)tokens.size()) {
        int n_chunk = std::min((int)tokens.size() - n_eval, (int)n_batch_size);
        batch.n_tokens = 0;
        for (int i = 0; i < n_chunk; ++i) {
            batch.token[i] = tokens[n_eval + i];
            batch.pos[i] = n_eval + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
            batch.n_tokens++;
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            env->ReleaseStringUTFChars(text, prompt);
            return nullptr;
        }
        n_eval += n_chunk;
    }

    float * embd = llama_get_embeddings(ctx);
    if (embd == nullptr) {
        llama_batch_free(batch);
        env->ReleaseStringUTFChars(text, prompt);
        return nullptr;
    }

    int n_embd = llama_model_n_embd(model);
    jdoubleArray result = env->NewDoubleArray(n_embd);
    std::vector<double> d_embd(n_embd);
    for(int i=0; i<n_embd; i++) d_embd[i] = (double)embd[i];
    env->SetDoubleArrayRegion(result, 0, n_embd, d_embd.data());

    llama_batch_free(batch);
    env->ReleaseStringUTFChars(text, prompt);

    // CRITICAL FIX: Clear the cache again after the embedding is done
    // so it doesn't corrupt the main chat generation!
    llama_memory_seq_rm(llama_get_memory(ctx), -1, -1, -1);

    return result;
}

// Accumulates raw token bytes across sendToken() calls so multi-byte UTF-8
// characters that llama.cpp splits across tokens (emoji, CJK, smart quotes) are
// only handed to Java once complete. Reset at the start of every generation.
static std::string g_token_utf8_buf;

// Length of the longest prefix of [s] that ends on a UTF-8 character boundary,
// i.e. excluding a trailing INCOMPLETE multi-byte sequence (the first bytes of a
// character whose remaining bytes are in the next token). Returns s.size() when
// the string already ends cleanly.
static size_t utf8_complete_prefix_len(const std::string &s) {
    const size_t n = s.size();
    if (n == 0) return 0;
    // Find the lead byte of the final sequence (skip back over 10xxxxxx bytes).
    size_t start = n - 1;
    while (start > 0 && ((unsigned char)s[start] & 0xC0) == 0x80) start--;
    const unsigned char lead = (unsigned char)s[start];
    size_t expected;
    if ((lead & 0x80) == 0x00) expected = 1;        // 0xxxxxxx
    else if ((lead & 0xE0) == 0xC0) expected = 2;   // 110xxxxx
    else if ((lead & 0xF0) == 0xE0) expected = 3;   // 1110xxxx
    else if ((lead & 0xF8) == 0xF0) expected = 4;   // 11110xxx
    else expected = 1;                              // invalid lead → emit as-is
    const size_t avail = n - start;
    return (avail >= expected) ? n : start; // hold back an incomplete final char
}

// Converts complete UTF-8 to UTF-16 (with surrogate pairs for astral chars like
// emoji). Needed because JNI NewStringUTF requires *Modified* UTF-8 and aborts on
// standard 4-byte sequences; NewString(jchar*) has no such restriction.
static std::vector<jchar> utf8_to_utf16(const std::string &s) {
    std::vector<jchar> out;
    out.reserve(s.size());
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp;
        size_t len;
        if ((c & 0x80) == 0x00) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { i++; continue; } // invalid lead byte → skip
        if (i + len > n) break; // safety (prefix should be complete)
        for (size_t k = 1; k < len; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        i += len;
        if (cp <= 0xFFFF) {
            out.push_back((jchar)cp);
        } else {
            cp -= 0x10000;
            out.push_back((jchar)(0xD800 + (cp >> 10)));
            out.push_back((jchar)(0xDC00 + (cp & 0x3FF)));
        }
    }
    return out;
}

static bool sendToken(JNIEnv *env, jobject thiz, jmethodID methodID, const struct llama_vocab * vocab, llama_token token, bool &is_eog_out) {
    char buf[128];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n <= 0) return false;
    std::string s(buf, n);
    if (s == "</s>" || s == "<|im_end|>" || s == "<|end|>") {
        is_eog_out = true;
        return false;
    }
    // Buffer, then emit only whole UTF-8 characters. Prevents the JNI abort
    // ("input is not valid Modified UTF-8") when an emoji/multi-byte char is
    // split across tokens or uses a 4-byte sequence.
    g_token_utf8_buf += s;
    const size_t emitLen = utf8_complete_prefix_len(g_token_utf8_buf);
    if (emitLen == 0) return true; // nothing complete yet — wait for more bytes
    const std::string piece = g_token_utf8_buf.substr(0, emitLen);
    g_token_utf8_buf.erase(0, emitLen);
    const std::vector<jchar> u16 = utf8_to_utf16(piece);
    jstring js = env->NewString(u16.data(), (jsize)u16.size());
    env->CallVoidMethod(thiz, methodID, js);
    env->DeleteLocalRef(js);
    return true;
}

JNIEXPORT void JNICALL
Java_com_timebox_native_1llama_NativeLlamaPlugin_startNativeGeneration(JNIEnv *env, jobject thiz, jobjectArray roles, jobjectArray contents, jobjectArray media_paths, jfloat temperature, jint top_k, jfloat top_p, jfloat repeat_penalty, jint penalty_last_n, jfloat freq_penalty, jfloat presence_penalty, jstring grammar) {
if (ctx == nullptr || model == nullptr) return;

stop_generation = false;
g_token_utf8_buf.clear(); // fresh UTF-8 reassembly buffer per generation
jclass clazz = env->GetObjectClass(thiz);
jmethodID methodID = env->GetMethodID(clazz, "onTokenReceived", "(Ljava/lang/String;)V");

const struct llama_vocab * vocab = llama_model_get_vocab(model);

int n_media = media_paths != nullptr ? env->GetArrayLength(media_paths) : 0;
bool use_draft = (ctx_draft != nullptr && n_media == 0);

// Fully reset context between generations. A partial seq_rm(-1,-1,-1) does NOT
// reset the cell tails of recurrent/hybrid models (e.g. Qwen3-Next / Mamba),
// tripping GGML_ASSERT(cell.has_seq_id(seq_id)) in find_slot on the next decode.
llama_memory_clear(llama_get_memory(ctx), true);
if (use_draft) llama_memory_clear(llama_get_memory(ctx_draft), true);

uint32_t n_batch_size = llama_n_batch(ctx);
if (use_draft) { n_batch_size = std::min(n_batch_size, llama_n_batch(ctx_draft)); }

int n_msg = env->GetArrayLength(roles);
std::vector<llama_chat_message> chat(n_msg);

// Store the EXACT jstring references used for extraction
std::vector<jstring> stored_jroles(n_msg);
std::vector<jstring> stored_jcontents(n_msg);
std::vector<const char*> stored_croles(n_msg);
std::vector<const char*> stored_ccontents(n_msg);

for (int i = 0; i < n_msg; ++i) {
jstring jrole = (jstring)env->GetObjectArrayElement(roles, i);
jstring jcontent = (jstring)env->GetObjectArrayElement(contents, i);
const char* role_str = env->GetStringUTFChars(jrole, nullptr);
const char* content_str = env->GetStringUTFChars(jcontent, nullptr);

chat[i].role = role_str;
chat[i].content = content_str;

stored_jroles[i] = jrole;
stored_jcontents[i] = jcontent;
stored_croles[i] = role_str;
stored_ccontents[i] = content_str;
}

// Format with the model's OWN chat template via the Jinja (common_chat) path —
// the same one mtmd-cli uses. The C-API llama_chat_apply_template does NOT honor
// custom templates like NuExtract3's (which frame the extraction task + enable
// thinking) → hallucinated output. use_jinja=true fixes it. Falls back to a
// plain role/content concatenation if the template can't be applied.
std::string prompt_str;
try {
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    // Thinking ON: NuExtract3 appears to need its reasoning step to actually
    // READ the page (accurate only with thinking on in testing). The Dart layer
    // bounds runaway generation with a hard token cap + stop-on-JSON.
    inputs.enable_thinking = true;
    // Assistant prefill: a trailing message with role "assistant" is NOT a
    // completed turn — it's a seed the model must CONTINUE (e.g. "{" to force an
    // immediate JSON reply from a model that would otherwise think out loud).
    // Rendering it through the template would close the turn with an
    // EOS/<|im_end|>, so instead we keep add_generation_prompt=true (prompt ends
    // at the open assistant turn) and append the seed text afterwards.
    int n_tmpl = n_msg;
    std::string assistant_prefix;
    if (n_msg > 0 && std::string(chat[n_msg - 1].role) == "assistant") {
        assistant_prefix = std::string(chat[n_msg - 1].content);
        n_tmpl -= 1; // don't feed the seed to the template
    }
    for (int i = 0; i < n_tmpl; ++i) {
        common_chat_msg m;
        m.role = chat[i].role;
        m.content = chat[i].content;
        inputs.messages.push_back(m);
    }
    common_chat_params cparams = common_chat_templates_apply(tmpls.get(), inputs);
    prompt_str = cparams.prompt + assistant_prefix;
} catch (const std::exception & e) {
    // fall through to fallback below
}
if (prompt_str.empty()) {
    for (int i = 0; i < n_msg; i++) {
        if (i + 1 == n_msg && std::string(chat[i].role) == "assistant") {
            // seed the assistant turn (prefill), leaving it open to continue
            prompt_str += "assistant: " + std::string(chat[i].content);
        } else {
            prompt_str += std::string(chat[i].role) + ": " + std::string(chat[i].content) + "\n";
        }
    }
    if (n_msg == 0 || std::string(chat[n_msg - 1].role) != "assistant") {
        prompt_str += "assistant: ";
    }
}

// RELEASE MEMORY AFTER THE PROMPT IS FULLY FORMATTED
for (int i = 0; i < n_msg; ++i) {
env->ReleaseStringUTFChars(stored_jroles[i], stored_croles[i]);
env->ReleaseStringUTFChars(stored_jcontents[i], stored_ccontents[i]);
env->DeleteLocalRef(stored_jroles[i]);
env->DeleteLocalRef(stored_jcontents[i]);
}

auto sparams = llama_sampler_chain_default_params();
llama_sampler * smpl = llama_sampler_chain_init(sparams);

// Optional GBNF grammar. Added FIRST so it masks every token the grammar
// forbids before temperature/top-k/top-p ever see the distribution; putting it
// later would let those samplers pick from candidates the grammar has already
// ruled out. With a grammar attached, malformed structured output stops being
// something the caller has to parse defensively — it becomes impossible.
//
// A grammar that fails to parse returns NULL. That is treated as "no grammar"
// rather than a fatal error: an unconstrained answer the caller can still
// validate beats refusing to generate at all.
if (grammar != nullptr) {
    const char * grammar_str = env->GetStringUTFChars(grammar, nullptr);
    if (grammar_str != nullptr && grammar_str[0] != '\0') {
        llama_sampler * gsmpl = llama_sampler_init_grammar(vocab, grammar_str, "root");
        if (gsmpl != nullptr) {
            llama_sampler_chain_add(smpl, gsmpl);
        } else {
            LOGE("Grammar failed to parse; continuing unconstrained");
        }
    }
    if (grammar_str != nullptr) env->ReleaseStringUTFChars(grammar, grammar_str);
}

llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
// Repetition penalty is caller-configurable: structured/list output (e.g. a JSON
// array of similar objects) is hurt by it — the repeated structural tokens get
// penalised, so the model emits fewer items and stops early. When the penalty is
// effectively off, skip the sampler entirely.
if (repeat_penalty > 1.0f || freq_penalty != 0.0f || presence_penalty != 0.0f) {
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(penalty_last_n, repeat_penalty, freq_penalty, presence_penalty));
}
llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

int n_cur = 0;
int n_prompt_tokens_total = 0;

// --- MTMD Evaluation ---
if (mtmd_ctx != nullptr && n_media > 0) {
std::string marker = mtmd_default_marker();
size_t pos = 0;
while ((pos = prompt_str.find("<|image_pad|>", pos)) != std::string::npos) {
prompt_str.replace(pos, 13, marker);
pos += marker.length();
}
pos = 0;
while ((pos = prompt_str.find("<|media_pad|>", pos)) != std::string::npos) {
prompt_str.replace(pos, 13, marker);
pos += marker.length();
}

std::vector<mtmd_bitmap *> bitmaps;
for (int i = 0; i < n_media; ++i) {
jstring jpath = (jstring)env->GetObjectArrayElement(media_paths, i);
const char* m_path = env->GetStringUTFChars(jpath, nullptr);

mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(mtmd_ctx, m_path);
if (bmp) bitmaps.push_back(bmp);

env->ReleaseStringUTFChars(jpath, m_path);
env->DeleteLocalRef(jpath);
}

mtmd_input_chunks * chunks = mtmd_input_chunks_init();
mtmd_input_text text_input = { prompt_str.c_str(), true, true };

int32_t tok_res = mtmd_tokenize(mtmd_ctx, chunks, &text_input, (const mtmd_bitmap **)bitmaps.data(), bitmaps.size());

if (tok_res == 0) {
llama_pos new_n_past = 0;

LOGI("Evaluating media tokens. This takes 10-30 seconds on Android CPU...");

// Only evaluates on Main Model context
mtmd_helper_eval_chunks(mtmd_ctx, ctx, chunks, 0, 0, n_batch_size, true, &new_n_past);

n_cur = new_n_past;
n_prompt_tokens_total = n_cur;
LOGI("Media evaluation complete! Context cursor is now at: %d", n_cur);
} else {
LOGE("MTMD Tokenize failed with error code: %d", tok_res);
n_prompt_tokens_total = 0;
}

mtmd_input_chunks_free(chunks);
for (auto b : bitmaps) mtmd_bitmap_free(b);

} else {
// --- STANDARD TEXT-ONLY FALLBACK ---
int n_prompt = -llama_tokenize(vocab, prompt_str.c_str(), prompt_str.length(), NULL, 0, true, true);
if (n_prompt < 0) n_prompt = -n_prompt;

std::vector<llama_token> prompt_tokens(n_prompt);
int tokenized_count = llama_tokenize(vocab, prompt_str.c_str(), prompt_str.length(), prompt_tokens.data(), prompt_tokens.size(), true, true);
prompt_tokens.resize(tokenized_count);

n_prompt_tokens_total = prompt_tokens.size();

llama_batch batch = llama_batch_init(n_batch_size, 0, 1);
int n_eval = 0;

while (n_eval < (int)prompt_tokens.size()) {
int n_chunk = std::min((int)prompt_tokens.size() - n_eval, (int)n_batch_size);
batch.n_tokens = 0;
for (int i = 0; i < n_chunk; ++i) {
batch.token[i] = prompt_tokens[n_eval + i];
batch.pos[i] = n_eval + i;
batch.n_seq_id[i] = 1;
batch.seq_id[i][0] = 0;
batch.logits[i] = (n_eval + i == prompt_tokens.size() - 1);
batch.n_tokens++;
}

if (llama_decode(ctx, batch) != 0) {
llama_batch_free(batch);
llama_sampler_free(smpl);
return;
}

if (use_draft) {
for (int i = 0; i < n_chunk; ++i) {
batch.logits[i] = (n_eval + i == prompt_tokens.size() - 1);
}
if (llama_decode(ctx_draft, batch) != 0) {
// Fail silently but safely disable draft mode for rest of session
use_draft = false;
}
}
n_eval += n_chunk;
}
n_cur = n_eval;
llama_batch_free(batch);
}

const uint32_t n_ctx_max = llama_n_ctx(ctx);
const int n_draft = 5;
bool is_eog_reached = false;

llama_batch decode_batch = llama_batch_init(1, 0, 1);
decode_batch.n_seq_id[0] = 1;
decode_batch.seq_id[0][0] = 0;

if (n_prompt_tokens_total <= 0) {
n_prompt_tokens_total = n_cur;
}

while (true) {
if (stop_generation || is_eog_reached) break;

if (n_cur + n_draft + 1 >= n_ctx_max) {
int n_keep = n_prompt_tokens_total;
if (n_keep >= n_ctx_max / 2) n_keep = n_ctx_max / 2;
const int n_discard = (n_ctx_max - n_keep) / 2;

llama_memory_seq_rm(llama_get_memory(ctx), 0, n_keep, n_keep + n_discard);
llama_memory_seq_add(llama_get_memory(ctx), 0, n_keep + n_discard, n_cur, -n_discard);
n_cur -= n_discard;

if (use_draft) {
llama_memory_seq_rm(llama_get_memory(ctx_draft), 0, n_keep, n_keep + n_discard);
llama_memory_seq_add(llama_get_memory(ctx_draft), 0, n_keep + n_discard, n_cur, -n_discard);
}
}

std::vector<llama_token> draft_tokens;
if (use_draft) {
for (int i = 0; i < n_draft; ++i) {
llama_token t = llama_sampler_sample(smpl, ctx_draft, -1);
llama_sampler_accept(smpl, t);
draft_tokens.push_back(t);

decode_batch.token[0] = draft_tokens.back();
decode_batch.pos[0] = n_cur + i;
decode_batch.n_tokens = 1;
decode_batch.logits[0] = true;

if (llama_decode(ctx_draft, decode_batch) != 0) { break; }
}
}

llama_token t_extra = llama_sampler_sample(smpl, ctx, -1);
int n_accepted = 0;

if (!draft_tokens.empty() && t_extra == draft_tokens[0]) {
sendToken(env, thiz, methodID, vocab, t_extra, is_eog_reached);
llama_sampler_accept(smpl, t_extra);
n_accepted = 1;

llama_batch b_tgt = llama_batch_init((int)draft_tokens.size(), 0, 1);
for (int i = 0; i < (int)draft_tokens.size(); ++i) {
b_tgt.token[i] = draft_tokens[i];
b_tgt.pos[i] = n_cur + i;
b_tgt.n_seq_id[i] = 1;
b_tgt.seq_id[i][0] = 0;
b_tgt.logits[i] = true;
}
b_tgt.n_tokens = (int)draft_tokens.size();
if (llama_decode(ctx, b_tgt) != 0) { llama_batch_free(b_tgt); break; }

for (int i = 1; i < (int)draft_tokens.size(); ++i) {
llama_token t_verified = llama_sampler_sample(smpl, ctx, i - 1);
if (t_verified == draft_tokens[i]) {
sendToken(env, thiz, methodID, vocab, t_verified, is_eog_reached);
llama_sampler_accept(smpl, t_verified);
n_accepted++;
if (is_eog_reached || llama_vocab_is_eog(vocab, t_verified)) { is_eog_reached = true; break; }
} else {
t_extra = t_verified;
break;
}
}
if (!is_eog_reached && n_accepted == (int)draft_tokens.size()) {
t_extra = llama_sampler_sample(smpl, ctx, (int)draft_tokens.size() - 1);
}
if (n_accepted < (int)draft_tokens.size()) {
llama_memory_seq_rm(llama_get_memory(ctx), 0, n_cur + n_accepted, -1);
if (use_draft) llama_memory_seq_rm(llama_get_memory(ctx_draft), 0, n_cur + n_accepted, -1);
}
llama_batch_free(b_tgt);
}

if (is_eog_reached) break;

sendToken(env, thiz, methodID, vocab, t_extra, is_eog_reached);
llama_sampler_accept(smpl, t_extra);
if (is_eog_reached || llama_vocab_is_eog(vocab, t_extra)) { is_eog_reached = true; break; }

decode_batch.token[0] = t_extra;
decode_batch.pos[0] = n_cur + n_accepted;
decode_batch.n_tokens = 1;
decode_batch.logits[0] = true;

if (llama_decode(ctx, decode_batch) != 0) { break; }
if (use_draft) llama_decode(ctx_draft, decode_batch);

n_cur += n_accepted + 1;
}

jstring eos = env->NewStringUTF("__END_OF_STREAM__");
env->CallVoidMethod(thiz, methodID, eos);
env->DeleteLocalRef(eos);

llama_batch_free(decode_batch);
llama_sampler_free(smpl);

std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

JNIEXPORT void JNICALL
Java_com_timebox_native_1llama_NativeLlamaPlugin_abortGeneration(JNIEnv *env, jobject thiz) {
stop_generation = true;
}

JNIEXPORT void JNICALL
Java_com_timebox_native_1llama_NativeLlamaPlugin_disposeLlama(JNIEnv *env, jobject thiz) {
    if (ctx) { llama_free(ctx); ctx = nullptr; }
    if (model) { llama_model_free(model); model = nullptr; }
    if (ctx_draft) { llama_free(ctx_draft); ctx_draft = nullptr; }
    if (model_draft) { llama_model_free(model_draft); model_draft = nullptr; }

    if (mtmd_ctx) { mtmd_free(mtmd_ctx); mtmd_ctx = nullptr; }

    llama_backend_free();
}

JNIEXPORT jint JNICALL
Java_com_timebox_native_1llama_NativeLlamaPlugin_getCpuCores(JNIEnv *env, jobject thiz, jboolean performance_only) {
    if (performance_only) {
        return getPerformanceCores();
    }
    return (jint)sysconf(_SC_NPROCESSORS_CONF);
}

} // extern "C"