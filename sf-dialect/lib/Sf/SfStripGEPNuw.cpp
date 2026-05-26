//===- SfStripGEPNuw.cpp - Strip nuw flag from LLVM GEP ops ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LLVM 22's MLIR LLVM dialect may emit a `nuw` (no unsigned wrap) flag on
// `llvm.getelementptr` operations.  Older LLVM versions (e.g. LLVM 20 used
// by mlir-translate in this project) cannot parse this flag.  This pass
// strips the `nuw` flag while preserving all other no-wrap flags.
//
// This replaces a regex-based fixup in `_fixup_mlir_for_translate` that
// operated on the MLIR text after lowering.
//
//===----------------------------------------------------------------------===//

#include "Sf/SfPasses.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DEF_SFSTRIPGEPNUW
#include "Sf/SfPasses.h.inc"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "sf-strip-gep-nuw"

using namespace mlir;

namespace {

struct SfStripGEPNuwPass
    : public ::impl::SfStripGEPNuwBase<SfStripGEPNuwPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfStripGEPNuwPass)

  void runOnOperation() override {
    bool modified = false;
    this->getOperation()->walk([&](LLVM::GEPOp op) {
      auto flags = op.getNoWrapFlags();
      if (bitEnumContainsAny(flags, LLVM::GEPNoWrapFlags::nuw)) {
        op.setNoWrapFlags(bitEnumClear(flags, LLVM::GEPNoWrapFlags::nuw));
        modified = true;
      }
    });

    if (modified) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[sf-strip-gep-nuw] stripped nuw flags from "
                 << "llvm.getelementptr ops\n");
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::sf::createSfStripGEPNuw() {
  return std::make_unique<SfStripGEPNuwPass>();
}
