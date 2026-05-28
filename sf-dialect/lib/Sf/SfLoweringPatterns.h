#pragma once

namespace mlir {
class RewritePatternSet;
namespace sf {
void registerActivationPatterns(RewritePatternSet &patterns);
void registerMatmulPatterns(RewritePatternSet &patterns);
void registerShapePatterns(RewritePatternSet &patterns);
void registerAttentionPatterns(RewritePatternSet &patterns);
void registerNormalizationPatterns(RewritePatternSet &patterns);
void registerGenOpsPatterns(RewritePatternSet &patterns);
void registerSeqOpsPatterns(RewritePatternSet &patterns);
void registerComparePatterns(RewritePatternSet &patterns);
void registerReducePatterns(RewritePatternSet &patterns);
void registerFusedPatterns(RewritePatternSet &patterns);
}
}
