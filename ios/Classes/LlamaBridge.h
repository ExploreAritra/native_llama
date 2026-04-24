#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface LlamaBridge : NSObject

+ (instancetype)shared;

// Updated signatures with overrides
- (BOOL)initModel:(NSString *)modelPath nCtx:(int)nCtx nThreads:(int)nThreads;
- (BOOL)initDraftModel:(NSString *)modelPath nCtx:(int)nCtx nThreads:(int)nThreads;
- (NSArray<NSNumber *> *)getEmbedding:(NSString *)text;

// Updated signature for dynamic samplers
- (void)startGenerationWithRoles:(NSArray<NSString *> *)roles
        contents:(NSArray<NSString *> *)contents
        temperature:(float)temperature
        topK:(int)topK
        topP:(float)topP
        onToken:(void (^)(NSString * _Nullable))onToken;

- (void)abortGeneration;
- (void)dispose;

@end

NS_ASSUME_NONNULL_END