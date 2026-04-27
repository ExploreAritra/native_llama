#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface LlamaBridge : NSObject

+ (instancetype)shared;

- (BOOL)initModel:(NSString *)modelPath nCtx:(int)nCtx nThreads:(int)nThreads nGpuLayers:(int)nGpuLayers;
- (BOOL)initDraftModel:(NSString *)modelPath nCtx:(int)nCtx nThreads:(int)nThreads nGpuLayers:(int)nGpuLayers;
- (BOOL)initVision:(NSString *)mmprojPath; // NEW
- (NSArray<NSNumber *> *)getEmbedding:(NSString *)text;

// Updated signature with imagePaths array
- (void)startGenerationWithRoles:(NSArray<NSString *> *)roles
        contents:(NSArray<NSString *> *)contents
        mediaPaths:(NSArray<NSString *> * _Nullable)mediaPaths
        temperature:(float)temperature
        topK:(int)topK
        topP:(float)topP
        onToken:(void (^)(NSString * _Nullable))onToken;

- (int)getCpuCores:(BOOL)performanceOnly;

- (void)abortGeneration;
- (void)dispose;

@end

NS_ASSUME_NONNULL_END