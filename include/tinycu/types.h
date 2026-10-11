#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

namespace tinycu {

/// A semantic type. Types are interned by TypeContext, so two types are equal iff their
/// pointers are equal.
class Type {
public:
  enum class Kind {
    Error, ///< result of an ill-typed expression; suppresses cascading diagnostics
    Void,
    Bool,        ///< internal: comparisons and logical operators
    Int,         ///< s32
    Float,       ///< f32
    Ptr,         ///< `int*` / `float*` in global memory
    SharedArray, ///< `shared int[N]` / `shared float[N]`
  };

  Kind kind() const { return kind_; }
  /// Element type of Ptr / SharedArray, otherwise null.
  const Type *elem() const { return elem_; }
  /// Number of elements of a SharedArray, otherwise 0.
  int32_t arraySize() const { return size_; }

  bool isError() const { return kind_ == Kind::Error; }
  bool isBool() const { return kind_ == Kind::Bool; }
  bool isInt() const { return kind_ == Kind::Int; }
  bool isFloat() const { return kind_ == Kind::Float; }
  bool isArithmetic() const { return isInt() || isFloat(); }
  bool isPtr() const { return kind_ == Kind::Ptr; }
  bool isSharedArray() const { return kind_ == Kind::SharedArray; }
  /// Can be indexed with `[]`.
  bool isIndexable() const { return isPtr() || isSharedArray(); }

  /// Source-like spelling: `int`, `float*`, `shared float[256]`, `bool`, `<error>`.
  std::string str() const;

private:
  friend class TypeContext;
  Type(Kind k, const Type *e, int32_t n) : kind_(k), elem_(e), size_(n) {}
  Kind kind_;
  const Type *elem_;
  int32_t size_;
};

/// Owns and interns all types. A single process-wide instance (`TypeContext::get()`)
/// keeps the API simple; types are immutable so sharing is safe.
class TypeContext {
public:
  static TypeContext &get();

  const Type *errorTy() const { return &error_; }
  const Type *voidTy() const { return &void_; }
  const Type *boolTy() const { return &bool_; }
  const Type *intTy() const { return &int_; }
  const Type *floatTy() const { return &float_; }
  const Type *ptrTo(const Type *elem) const { return elem->isInt() ? &intPtr_ : &floatPtr_; }
  const Type *sharedArray(const Type *elem, int32_t n);

private:
  TypeContext();
  Type error_, void_, bool_, int_, float_, intPtr_, floatPtr_;
  std::map<std::pair<const Type *, int32_t>, std::unique_ptr<Type>> shared_;
};

} // namespace tinycu
