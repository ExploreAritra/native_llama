#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
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

#define TAG "NATIVE_LLAMA_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model * model = nullptr;
static llama_context * ctx = nullptr;
static llama_model * model_draft = nullptr;
static llama_context * ctx_draft = nullptr;
static mtmd_context * mtmd_ctx = nullptr;
static bool stop_generation = false;

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

    llama_backend_init();
    ggml_backend_load_all();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    model = llama_model_load_from_file(path, mparams);

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
    if (ctx == nullptr) {
        LOGE("Failed to create context");
        llama_model_free(model);
        model = nullptr;
        env->ReleaseStringUTFChars(model_path, path);
        return JNI_FALSE;
    }

    env->ReleaseStringUTFChars(model_path, path);
    LOGI("Llama model initialized successfully");
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
    mtmd_params.use_gpu = true;
    mtmd_params.image_max_tokens = 1024;
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

static bool sendToken(JNIEnv *env, jobject thiz, jmethodID methodID, const struct llama_vocab * vocab, llama_token token, bool &is_eog_out) {
    char buf[128];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n > 0) {
        std::string s(buf, n);
        if (s == "</s>" || s == "<|im_end|>" || s == "<|end|>") {
            is_eog_out = true;
            return false;
        }
        jstring js = env->NewStringUTF(s.c_str());
        env->CallVoidMethod(thiz, methodID, js);
        env->DeleteLocalRef(js);
        return true;
    }
    return false;
}

JNIEXPORT void JNICALL
Java_com_timebox_native_1llama_NativeLlamaPlugin_startNativeGeneration(JNIEnv *env, jobject thiz, jobjectArray roles, jobjectArray contents, jobjectArray media_paths, jfloat temperature, jint top_k, jfloat top_p) {
if (ctx == nullptr || model == nullptr) return;

stop_generation = false;
jclass clazz = env->GetObjectClass(thiz);
jmethodID methodID = env->GetMethodID(clazz, "onTokenReceived", "(Ljava/lang/String;)V");

const struct llama_vocab * vocab = llama_model_get_vocab(model);

int n_media = media_paths != nullptr ? env->GetArrayLength(media_paths) : 0;
bool use_draft = (ctx_draft != nullptr && n_media == 0);

llama_memory_seq_rm(llama_get_memory(ctx), -1, -1, -1);
if (use_draft) llama_memory_seq_rm(llama_get_memory(ctx_draft), -1, -1, -1);

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

char tmpl[2048];
int32_t tmpl_len = llama_model_meta_val_str(model, "tokenizer.chat_template", tmpl, sizeof(tmpl));
const char* tmpl_ptr = (tmpl_len > 0) ? tmpl : nullptr;

int32_t n_formatted = llama_chat_apply_template(tmpl_ptr, chat.data(), n_msg, true, nullptr, 0);
std::vector<char> formatted_prompt;
if (n_formatted > 0) {
formatted_prompt.resize(n_formatted + 1);
llama_chat_apply_template(tmpl_ptr, chat.data(), n_msg, true, formatted_prompt.data(), formatted_prompt.size());
} else {
std::string s = "";
for(int i=0; i<n_msg; i++) { s += std::string(chat[i].role) + ": " + std::string(chat[i].content) + "\n"; }
s += "assistant: ";
formatted_prompt.assign(s.begin(), s.end());
formatted_prompt.push_back('\0');
n_formatted = s.length();
}

std::string prompt_str = formatted_prompt.data();

// RELEASE MEMORY AFTER THE PROMPT IS FULLY FORMATTED
for (int i = 0; i < n_msg; ++i) {
env->ReleaseStringUTFChars(stored_jroles[i], stored_croles[i]);
env->ReleaseStringUTFChars(stored_jcontents[i], stored_ccontents[i]);
env->DeleteLocalRef(stored_jroles[i]);
env->DeleteLocalRef(stored_jcontents[i]);
}

auto sparams = llama_sampler_chain_default_params();
llama_sampler * smpl = llama_sampler_chain_init(sparams);
llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_penalties(128, 1.2f, 0.1f, 0.1f));
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