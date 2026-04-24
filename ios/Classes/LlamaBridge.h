#import <Foundation/Foundation.h>

@interface LlamaBridge : NSObject

+ (instancetype)shared;
- (BOOL)initModel:(NSString *)modelPath;
- (BOOL)initDraftModel:(NSString *)draftModelPath;
- (NSArray<NSNumber *> *)getEmbedding:(NSString *)text;
- (void)startGenerationWithRoles:(NSArray<NSString *> *)roles contents:(NSArray<NSString *> *)contents onToken:(void (^)(NSString *))onToken;
- (void)abortGeneration;
- (void)dispose;
- (void)unload;

@end