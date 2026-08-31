#include <gtest/gtest.h>
#include "Sf/ViewShapeSolver.h"

using namespace mlir::sf;

TEST(ViewShapeSolver, AllStatic) {
    auto plan = resolveViewShape({4, 12, 64}, 0);
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan[0].kind, DimSourceKind::Static);
    EXPECT_EQ(*plan[0].staticVal, 4);
    EXPECT_EQ(plan[1].kind, DimSourceKind::Static);
    EXPECT_EQ(*plan[1].staticVal, 12);
    EXPECT_EQ(plan[2].kind, DimSourceKind::Static);
    EXPECT_EQ(*plan[2].staticVal, 64);
}

TEST(ViewShapeSolver, MixedSentinelAndStatic) {
    auto plan = resolveViewShape({-2, -3, 12, 64}, 2);
    ASSERT_EQ(plan.size(), 4u);
    EXPECT_EQ(plan[0].kind, DimSourceKind::DynOperand);
    EXPECT_EQ(*plan[0].operandIdx, 0);
    EXPECT_EQ(plan[1].kind, DimSourceKind::DynOperand);
    EXPECT_EQ(*plan[1].operandIdx, 1);
    EXPECT_EQ(plan[2].kind, DimSourceKind::Static);
    EXPECT_EQ(plan[3].kind, DimSourceKind::Static);
}

TEST(ViewShapeSolver, InferredLastDim) {
    auto plan = resolveViewShape({-2, -3, -1}, 2);
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan[0].kind, DimSourceKind::DynOperand);
    EXPECT_EQ(*plan[0].operandIdx, 0);
    EXPECT_EQ(plan[1].kind, DimSourceKind::DynOperand);
    EXPECT_EQ(*plan[1].operandIdx, 1);
    EXPECT_EQ(plan[2].kind, DimSourceKind::Inferred);
}

TEST(ViewShapeSolver, Neg1WithExtraOperands) {
    auto plan = resolveViewShape({-2, -1}, 2);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0].kind, DimSourceKind::DynOperand);
    EXPECT_EQ(*plan[0].operandIdx, 0);
    EXPECT_EQ(plan[1].kind, DimSourceKind::DynOperand);
    EXPECT_EQ(*plan[1].operandIdx, 1);
}

TEST(ViewShapeSolver, OnlyInferred) {
    auto plan = resolveViewShape({-1, -1}, 0);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0].kind, DimSourceKind::Inferred);
    EXPECT_EQ(plan[1].kind, DimSourceKind::Inferred);
}

TEST(ViewShapeSolver, AllExplicitSentinels) {
    auto plan = resolveViewShape({-2, -3, -4}, 3);
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan[0].kind, DimSourceKind::DynOperand);
    EXPECT_EQ(*plan[0].operandIdx, 0);
    EXPECT_EQ(plan[1].kind, DimSourceKind::DynOperand);
    EXPECT_EQ(*plan[1].operandIdx, 1);
    EXPECT_EQ(plan[2].kind, DimSourceKind::DynOperand);
    EXPECT_EQ(*plan[2].operandIdx, 2);
}

TEST(ViewShapeSolver, EmptyShapeAttr) {
    auto plan = resolveViewShape({}, 0);
    EXPECT_TRUE(plan.empty());
}

TEST(ViewShapeSolver, SSARefConsumesOperand) {
    auto plan = resolveViewShape({kSSARefSentinel, -1}, 1);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0].kind, DimSourceKind::DynOperand);
    EXPECT_EQ(*plan[0].operandIdx, 0);
    EXPECT_EQ(plan[1].kind, DimSourceKind::Inferred);
}

TEST(ViewShapeSolver, SSARefInsufficientOperands) {
    auto plan = resolveViewShape({kSSARefSentinel}, 0);
    EXPECT_TRUE(plan.empty());
}

TEST(ViewShapeSolver, MixedNegSentinelAndSSARef) {
    auto plan = resolveViewShape({-2, kSSARefSentinel, -1}, 2);
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan[0].kind, DimSourceKind::DynOperand);
    EXPECT_EQ(*plan[0].operandIdx, 0);
    EXPECT_EQ(plan[1].kind, DimSourceKind::DynOperand);
    EXPECT_EQ(*plan[1].operandIdx, 1);
    EXPECT_EQ(plan[2].kind, DimSourceKind::Inferred);
}
