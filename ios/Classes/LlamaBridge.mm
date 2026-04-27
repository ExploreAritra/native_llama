#import "LlamaBridge.h"
#import <llama.h>
#import "ggml-backend.h"
#import <vector>
#import <string>
#import <sys/stat.h>

// --- MTMD Headers ---
#import "mtmd.h"
#import "mtmd-helper.h"

@implementation LlamaBridge {
    llama_model *model;
    llama_context *ctx;
    llama_model *model_draft;
    llama_context *ctx_draft;
    mtmd_context *mtmd_ctx;
    bool stop_generation;
}

+ (instancetype)shared {
    static LlamaBridge *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] init];
    });
    return sharedInstance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        llama_backend_init();
        ggml_backend_load_all();
        model = nullptr;
        ctx = nullptr;
        model_draft = nullptr;
        ctx_draft = nullptr;
        mtmd_ctx = nullptr;
        stop_generation = false;
    }
    return self;
}

- (BOOL)initModel:(NSString *)modelPath nCtx:(int)nCtx nThreads:(int)nThreads nGpuLayers:(int)nGpuLayers {
    const char *path = [modelPath UTF8String];

    uint64_t physicalMemory = [[NSProcessInfo processInfo] physicalMemory];
    double memoryGB = physicalMemory / (1024.0 * 1024.0 * 1024.0);

    struct stat stat_buf;
    double fileSizeGB = 0;
    if (stat(path, &stat_buf) == 0) {
        fileSizeGB = stat_buf.st_size / (1024.0 * 1024.0 * 1024.0);
    }

    if (fileSizeGB > 0 && fileSizeGB > (memoryGB * 0.65)) {
        NSLog(@"NATIVE_LLAMA: RAM SHIELD - Model size (%.2f GB) exceeds safe limits for device RAM (%.2f GB). Aborting load.", fileSizeGB, memoryGB);
        return NO;
    }

    [self unload];

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = nGpuLayers;
    mparams.use_mmap = false;

    model = llama_model_load_from_file(path, mparams);
    if (model == nullptr) return NO;

    auto cparams = llama_context_default_params();
    cparams.n_threads = nThreads > 0 ? nThreads : 4;
    cparams.embeddings = true;
    cparams.type_k = GGML_TYPE_Q8_0;
    cparams.type_v = GGML_TYPE_Q8_0;
    cparams.n_batch = 128;

    if (nCtx > 0) {
        cparams.n_ctx = nCtx;
    } else {
        int32_t dynamic_n_ctx = 4096;
        if (memoryGB >= 7.5) dynamic_n_ctx = 8192;
        if (memoryGB >= 11.5) dynamic_n_ctx = 16384;
        cparams.n_ctx = dynamic_n_ctx;
    }

    ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        model = nullptr;
        return NO;
    }
    return YES;
}

- (BOOL)initVision:(NSString *)mmprojPath {
    const char *path = [mmprojPath UTF8String];
    NSLog(@"Initializing Vision model from: %s", path);

    if (mtmd_ctx) { mtmd_free(mtmd_ctx); mtmd_ctx = nullptr; }

    if (model == nullptr) {
        NSLog(@"Error: Main text model must be initialized before the vision projector.");
        return NO;
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = true;

    mtmd_ctx = mtmd_init_from_file(path, model, mtmd_params);

    if (mtmd_ctx == nullptr) {
        NSLog(@"Failed to load MTMD mmproj model: %s", path);
        return NO;
    }

    NSLog(@"Vision model initialized successfully via MTMD");
    return YES;
}

- (BOOL)initDraftModel:(NSString *)modelPath nCtx:(int)nCtx nThreads:(int)nThreads nGpuLayers:(int)nGpuLayers {
    const char *path = [modelPath UTF8String];
    if (ctx_draft) { llama_free(ctx_draft); ctx_draft = nullptr; }
    if (model_draft) { llama_model_free(model_draft); model_draft = nullptr; }

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = nGpuLayers;
    mparams.use_mmap = false;

    model_draft = llama_model_load_from_file(path, mparams);
    if (model_draft == nullptr) return NO;

    auto cparams = llama_context_default_params();
    cparams.n_threads = nThreads > 0 ? nThreads : 4;
    cparams.embeddings = true;
    cparams.type_k = GGML_TYPE_Q8_0;
    cparams.type_v = GGML_TYPE_Q8_0;
    cparams.n_batch = 32;

    if (nCtx > 0) {
        cparams.n_ctx = nCtx;
    } else {
        uint64_t physicalMemory = [[NSProcessInfo processInfo] physicalMemory];
        double memoryGB = physicalMemory / (1024.0 * 1024.0 * 1024.0);
        int32_t dynamic_n_ctx = 4096;
        if (memoryGB >= 7.5) dynamic_n_ctx = 8192;
        if (memoryGB >= 11.5) dynamic_n_ctx = 16384;
        cparams.n_ctx = dynamic_n_ctx;
    }

    ctx_draft = llama_init_from_model(model_draft, cparams);
    return ctx_draft != nullptr;
}

- (NSArray<NSNumber *> *)getEmbedding:(NSString *)text {
    if (ctx == nullptr || model == nullptr) return nil;
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = [text UTF8String];

    int n_tokens = -llama_tokenize(vocab, prompt, (int)strlen(prompt), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt, (int)strlen(prompt), tokens.data(), (int)tokens.size(), true, true);

    uint32_t n_batch = llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    int n_eval = 0;
    while (n_eval < (int)tokens.size()) {
        int n_chunk = MIN((int)tokens.size() - n_eval, n_batch);
        batch.n_tokens = 0;
        for (int i = 0; i < n_chunk; ++i) {
            batch.token[i] = tokens[n_eval + i];
            batch.pos[i] = n_eval + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
            batch.n_tokens++;
        }
        if (llama_decode(ctx, batch) != 0) { llama_batch_free(batch); return nil; }
        n_eval += n_chunk;
    }

    float * embd = llama_get_embeddings(ctx);
    if (embd == nullptr) { llama_batch_free(batch); return nil; }

    int n_embd = llama_model_n_embd(model);
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:n_embd];
    for (int i = 0; i < n_embd; i++) { [result addObject:@(embd[i])]; }

    llama_batch_free(batch);
    return result;
}

- (void)startGenerationWithRoles:(NSArray<NSString *> *)roles contents:(NSArray<NSString *> *)contents mediaPaths:(NSArray<NSString *> *)mediaPaths temperature:(float)temperature topK:(int)topK topP:(float)topP onToken:(void (^)(NSString *))onToken {
    if (ctx == nullptr || model == nullptr) return;
    stop_generation = false;
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // --- CRITICAL FIX: Only use the draft model if NO media is attached ---
    bool use_draft = (ctx_draft != nullptr && (mediaPaths == nil || mediaPaths.count == 0));

    llama_memory_seq_rm(llama_get_memory(ctx), -1, -1, -1);
    if (use_draft) llama_memory_seq_rm(llama_get_memory(ctx_draft), -1, -1, -1);

    uint32_t n_batch_size = llama_n_batch(ctx);
    if (use_draft) { n_batch_size = MIN(n_batch_size, llama_n_batch(ctx_draft)); }

    std::vector<llama_chat_message> chat;
    for (NSUInteger i = 0; i < roles.count; i++) {
        chat.push_back({
                               .role = [roles[i] UTF8String],
                               .content = [contents[i] UTF8String]
                       });
    }

    char tmpl[2048];
    int32_t tmpl_len = llama_model_meta_val_str(model, "tokenizer.chat_template", tmpl, sizeof(tmpl));
    const char* tmpl_ptr = (tmpl_len > 0) ? tmpl : nullptr;

    int32_t n_formatted = llama_chat_apply_template(tmpl_ptr, chat.data(), chat.size(), true, nullptr, 0);
    std::vector<char> formatted_prompt;
    if (n_formatted > 0) {
        formatted_prompt.resize(n_formatted + 1);
        llama_chat_apply_template(tmpl_ptr, chat.data(), chat.size(), true, formatted_prompt.data(), formatted_prompt.size());
    } else {
        std::string s = "";
        for (auto &msg : chat) {
            s += std::string(msg.role) + ": " + std::string(msg.content) + "\n";
        }
        s += "assistant: ";
        formatted_prompt.assign(s.begin(), s.end());
        formatted_prompt.push_back('\0');
        n_formatted = (int32_t)s.length();
    }

    std::string prompt_str = formatted_prompt.data();

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(topK));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(128, 1.2f, 0.1f, 0.1f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    int n_cur = 0;
    int n_prompt_tokens_total = 0;

    // --- MTMD Evaluation ---
    if (mtmd_ctx != nullptr && mediaPaths != nullptr && mediaPaths.count > 0) {

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
        for (NSString *mediaPath in mediaPaths) {
            mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(mtmd_ctx, [mediaPath UTF8String]);
            if (bmp) bitmaps.push_back(bmp);
        }

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        mtmd_input_text text_input = { prompt_str.c_str(), true, true };

        int32_t tok_res = mtmd_tokenize(mtmd_ctx, chunks, &text_input, (const mtmd_bitmap **)bitmaps.data(), bitmaps.size());

        if (tok_res == 0) {
            llama_pos new_n_past = 0;

            // Only evaluates on Main Model context (Draft is skipped safely)
            mtmd_helper_eval_chunks(mtmd_ctx, ctx, chunks, 0, 0, n_batch_size, true, &new_n_past);

            // --- CRITICAL FIX: Use the actual cursor position returned by the engine ---
            n_cur = new_n_past;
            n_prompt_tokens_total = n_cur;
        } else {
            NSLog(@"MTMD Tokenize failed with error code: %d", tok_res);
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
            int n_chunk = MIN((int)prompt_tokens.size() - n_eval, n_batch_size);
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
                    use_draft = false; // Fail safely
                }
            }
            n_eval += n_chunk;
        }
        n_cur = n_eval;
        llama_batch_free(batch);
    }
    // -----------------------------------------

    const uint32_t n_ctx_max = llama_n_ctx(ctx);
    const int n_draft = 5;
    bool is_eog_reached = false;

    llama_batch decode_batch = llama_batch_init(1, 0, 1);
    decode_batch.n_seq_id[0] = 1;
    decode_batch.seq_id[0][0] = 0;

    // --- CRITICAL FIX: Safe fallback for token count ---
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
            is_eog_reached = [self _sendToken:t_extra vocab:vocab onToken:onToken];
            llama_sampler_accept(smpl, t_extra);
            n_accepted = 1;

            if (is_eog_reached || llama_vocab_is_eog(vocab, t_extra)) {
                is_eog_reached = true;
            } else {
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
                        if ([self _sendToken:t_verified vocab:vocab onToken:onToken]) { is_eog_reached = true; }
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
        }

        if (is_eog_reached) break;

        if ([self _sendToken:t_extra vocab:vocab onToken:onToken]) { is_eog_reached = true; }
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

    onToken(@"__END_OF_STREAM__");
    llama_batch_free(decode_batch);
    llama_sampler_free(smpl);
}

- (void)abortGeneration {
    stop_generation = true;
}

- (void)unload {
    if (ctx) { llama_free(ctx); ctx = nullptr; }
    if (model) { llama_model_free(model); model = nullptr; }
    if (ctx_draft) { llama_free(ctx_draft); ctx_draft = nullptr; }
    if (model_draft) { llama_model_free(model_draft); model_draft = nullptr; }
    if (mtmd_ctx) { mtmd_free(mtmd_ctx); mtmd_ctx = nullptr; }
}

- (void)dispose {
    [self unload];
    llama_backend_free();
}

- (int)getCpuCores:(BOOL)performanceOnly {
    if (performanceOnly) {
        // iOS doesn't easily expose performance cores vs efficiency cores in a standard way
        // like Android's sysfs, but usually 2 or 4 is a safe bet for high performance cores.
        // We'll return active processor count as a fallback.
        int cores = (int)[[NSProcessInfo processInfo] activeProcessorCount];
        return cores > 2 ? cores / 2 : cores;
    }
    return (int)[[NSProcessInfo processInfo] activeProcessorCount];
}

- (BOOL)_sendToken:(llama_token)token vocab:(const struct llama_vocab *)vocab onToken:(void (^)(NSString *))onToken {
    char buf[128];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n > 0) {
        std::string s(buf, n);
        if (s == "</s>" || s == "<|im_end|>" || s == "<|end|>") return YES;
        onToken([NSString stringWithUTF8String:s.c_str()]);
    }
    return NO;
}

@end