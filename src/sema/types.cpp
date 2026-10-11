#include "tinycu/types.h"

namespace tinycu {

std::string Type::str() const {
  switch (kind_) {
  case Kind::Error:
    return "<error>";
  case Kind::Void:
    return "void";
  case Kind::Bool:
    return "bool";
  case Kind::Int:
    return "int";
  case Kind::Float:
    return "float";
  case Kind::Ptr:
    return elem_->str() + "*";
  case Kind::SharedArray:
    return "shared " + elem_->str() + "[" + std::to_string(size_) + "]";
  }
  return "?";
}

TypeContext::TypeContext()
    : error_(Type::Kind::Error, nullptr, 0), void_(Type::Kind::Void, nullptr, 0),
      bool_(Type::Kind::Bool, nullptr, 0), int_(Type::Kind::Int, nullptr, 0),
      float_(Type::Kind::Float, nullptr, 0), intPtr_(Type::Kind::Ptr, &int_, 0),
      floatPtr_(Type::Kind::Ptr, &float_, 0) {}

TypeContext &TypeContext::get() {
  static TypeContext ctx;
  return ctx;
}

const Type *TypeContext::sharedArray(const Type *elem, int32_t n) {
  auto &slot = shared_[{elem, n}];
  if (!slot)
    slot.reset(new Type(Type::Kind::SharedArray, elem, n));
  return slot.get();
}

} // namespace tinycu
