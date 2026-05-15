//===-- SfOps.td - Python bindings for sf dialect ------*- tablegen -*-===//
//
// Part of the ServeForge Project.
// See LICENSE for license information.
//
//===------------------------------------------------------------------===//

#ifndef PYTHON_BINDINGS_SF_OPS
#define PYTHON_BINDINGS_SF_OPS

// Include dialect definition first so SfDialect.td can be found by the
// Python-binding tablegen generator which uses -Iinclude but not -Iinclude/Sf.
include "Sf/SfDialect.td"
include "Sf/SfOps.td"

#endif
