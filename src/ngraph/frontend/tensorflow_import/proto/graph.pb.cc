// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: graph.proto

#include "graph.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)
namespace tensorflow {
class GraphDefDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<GraphDef>
      _instance;
} _GraphDef_default_instance_;
}  // namespace tensorflow
namespace protobuf_graph_2eproto {
void InitDefaultsGraphDefImpl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
  ::google::protobuf::internal::InitProtobufDefaultsForceUnique();
#else
  ::google::protobuf::internal::InitProtobufDefaults();
#endif  // GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
  protobuf_node_5fdef_2eproto::InitDefaultsNodeDef();
  protobuf_versions_2eproto::InitDefaultsVersionDef();
  protobuf_function_2eproto::InitDefaultsFunctionDefLibrary();
  {
    void* ptr = &::tensorflow::_GraphDef_default_instance_;
    new (ptr) ::tensorflow::GraphDef();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::GraphDef::InitAsDefaultInstance();
}

void InitDefaultsGraphDef() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &InitDefaultsGraphDefImpl);
}

::google::protobuf::Metadata file_level_metadata[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::GraphDef, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::GraphDef, node_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::GraphDef, versions_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::GraphDef, version_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::GraphDef, library_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::tensorflow::GraphDef)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::tensorflow::_GraphDef_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  ::google::protobuf::MessageFactory* factory = NULL;
  AssignDescriptors(
      "graph.proto", schemas, file_default_instances, TableStruct::offsets, factory,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 1);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n\013graph.proto\022\ntensorflow\032\016node_def.prot"
      "o\032\016function.proto\032\016versions.proto\"\235\001\n\010Gr"
      "aphDef\022!\n\004node\030\001 \003(\0132\023.tensorflow.NodeDe"
      "f\022(\n\010versions\030\004 \001(\0132\026.tensorflow.Version"
      "Def\022\023\n\007version\030\003 \001(\005B\002\030\001\022/\n\007library\030\002 \001("
      "\0132\036.tensorflow.FunctionDefLibraryBk\n\030org"
      ".tensorflow.frameworkB\013GraphProtosP\001Z=gi"
      "thub.com/tensorflow/tensorflow/tensorflo"
      "w/go/core/framework\370\001\001b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 350);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "graph.proto", &protobuf_RegisterTypes);
  ::protobuf_node_5fdef_2eproto::AddDescriptors();
  ::protobuf_function_2eproto::AddDescriptors();
  ::protobuf_versions_2eproto::AddDescriptors();
}

void AddDescriptors() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_graph_2eproto
namespace tensorflow {

// ===================================================================

void GraphDef::InitAsDefaultInstance() {
  ::tensorflow::_GraphDef_default_instance_._instance.get_mutable()->versions_ = const_cast< ::tensorflow::VersionDef*>(
      ::tensorflow::VersionDef::internal_default_instance());
  ::tensorflow::_GraphDef_default_instance_._instance.get_mutable()->library_ = const_cast< ::tensorflow::FunctionDefLibrary*>(
      ::tensorflow::FunctionDefLibrary::internal_default_instance());
}
void GraphDef::clear_node() {
  node_.Clear();
}
void GraphDef::_slow_mutable_versions() {
  versions_ = ::google::protobuf::Arena::CreateMessage< ::tensorflow::VersionDef >(
      GetArenaNoVirtual());
}
void GraphDef::unsafe_arena_set_allocated_versions(
    ::tensorflow::VersionDef* versions) {
  if (GetArenaNoVirtual() == NULL) {
    delete versions_;
  }
  versions_ = versions;
  if (versions) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tensorflow.GraphDef.versions)
}
void GraphDef::clear_versions() {
  if (GetArenaNoVirtual() == NULL && versions_ != NULL) {
    delete versions_;
  }
  versions_ = NULL;
}
void GraphDef::_slow_mutable_library() {
  library_ = ::google::protobuf::Arena::CreateMessage< ::tensorflow::FunctionDefLibrary >(
      GetArenaNoVirtual());
}
void GraphDef::unsafe_arena_set_allocated_library(
    ::tensorflow::FunctionDefLibrary* library) {
  if (GetArenaNoVirtual() == NULL) {
    delete library_;
  }
  library_ = library;
  if (library) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tensorflow.GraphDef.library)
}
void GraphDef::clear_library() {
  if (GetArenaNoVirtual() == NULL && library_ != NULL) {
    delete library_;
  }
  library_ = NULL;
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int GraphDef::kNodeFieldNumber;
const int GraphDef::kVersionsFieldNumber;
const int GraphDef::kVersionFieldNumber;
const int GraphDef::kLibraryFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

GraphDef::GraphDef()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    ::protobuf_graph_2eproto::InitDefaultsGraphDef();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.GraphDef)
}
GraphDef::GraphDef(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena),
  node_(arena) {
  ::protobuf_graph_2eproto::InitDefaultsGraphDef();
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.GraphDef)
}
GraphDef::GraphDef(const GraphDef& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      node_(from.node_),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  if (from.has_library()) {
    library_ = new ::tensorflow::FunctionDefLibrary(*from.library_);
  } else {
    library_ = NULL;
  }
  if (from.has_versions()) {
    versions_ = new ::tensorflow::VersionDef(*from.versions_);
  } else {
    versions_ = NULL;
  }
  version_ = from.version_;
  // @@protoc_insertion_point(copy_constructor:tensorflow.GraphDef)
}

void GraphDef::SharedCtor() {
  ::memset(&library_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&version_) -
      reinterpret_cast<char*>(&library_)) + sizeof(version_));
  _cached_size_ = 0;
}

GraphDef::~GraphDef() {
  // @@protoc_insertion_point(destructor:tensorflow.GraphDef)
  SharedDtor();
}

void GraphDef::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
  if (this != internal_default_instance()) delete library_;
  if (this != internal_default_instance()) delete versions_;
}

void GraphDef::ArenaDtor(void* object) {
  GraphDef* _this = reinterpret_cast< GraphDef* >(object);
  (void)_this;
}
void GraphDef::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void GraphDef::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* GraphDef::descriptor() {
  ::protobuf_graph_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_graph_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const GraphDef& GraphDef::default_instance() {
  ::protobuf_graph_2eproto::InitDefaultsGraphDef();
  return *internal_default_instance();
}

GraphDef* GraphDef::New(::google::protobuf::Arena* arena) const {
  return ::google::protobuf::Arena::CreateMessage<GraphDef>(arena);
}

void GraphDef::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.GraphDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  node_.Clear();
  if (GetArenaNoVirtual() == NULL && library_ != NULL) {
    delete library_;
  }
  library_ = NULL;
  if (GetArenaNoVirtual() == NULL && versions_ != NULL) {
    delete versions_;
  }
  versions_ = NULL;
  version_ = 0;
  _internal_metadata_.Clear();
}

bool GraphDef::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.GraphDef)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .tensorflow.NodeDef node = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(input, add_node()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // .tensorflow.FunctionDefLibrary library = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
               input, mutable_library()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int32 version = 3 [deprecated = true];
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(24u /* 24 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &version_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // .tensorflow.VersionDef versions = 4;
      case 4: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(34u /* 34 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
               input, mutable_versions()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.GraphDef)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.GraphDef)
  return false;
#undef DO_
}

void GraphDef::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.GraphDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .tensorflow.NodeDef node = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->node_size()); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, this->node(static_cast<int>(i)), output);
  }

  // .tensorflow.FunctionDefLibrary library = 2;
  if (this->has_library()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, *this->library_, output);
  }

  // int32 version = 3 [deprecated = true];
  if (this->version() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(3, this->version(), output);
  }

  // .tensorflow.VersionDef versions = 4;
  if (this->has_versions()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      4, *this->versions_, output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:tensorflow.GraphDef)
}

::google::protobuf::uint8* GraphDef::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.GraphDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .tensorflow.NodeDef node = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->node_size()); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, this->node(static_cast<int>(i)), deterministic, target);
  }

  // .tensorflow.FunctionDefLibrary library = 2;
  if (this->has_library()) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        2, *this->library_, deterministic, target);
  }

  // int32 version = 3 [deprecated = true];
  if (this->version() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(3, this->version(), target);
  }

  // .tensorflow.VersionDef versions = 4;
  if (this->has_versions()) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        4, *this->versions_, deterministic, target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.GraphDef)
  return target;
}

size_t GraphDef::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.GraphDef)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // repeated .tensorflow.NodeDef node = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->node_size());
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSize(
          this->node(static_cast<int>(i)));
    }
  }

  // .tensorflow.FunctionDefLibrary library = 2;
  if (this->has_library()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSize(
        *this->library_);
  }

  // .tensorflow.VersionDef versions = 4;
  if (this->has_versions()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSize(
        *this->versions_);
  }

  // int32 version = 3 [deprecated = true];
  if (this->version() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->version());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void GraphDef::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.GraphDef)
  GOOGLE_DCHECK_NE(&from, this);
  const GraphDef* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const GraphDef>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.GraphDef)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.GraphDef)
    MergeFrom(*source);
  }
}

void GraphDef::MergeFrom(const GraphDef& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.GraphDef)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  node_.MergeFrom(from.node_);
  if (from.has_library()) {
    mutable_library()->::tensorflow::FunctionDefLibrary::MergeFrom(from.library());
  }
  if (from.has_versions()) {
    mutable_versions()->::tensorflow::VersionDef::MergeFrom(from.versions());
  }
  if (from.version() != 0) {
    set_version(from.version());
  }
}

void GraphDef::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.GraphDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void GraphDef::CopyFrom(const GraphDef& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.GraphDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool GraphDef::IsInitialized() const {
  return true;
}

void GraphDef::Swap(GraphDef* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    GraphDef* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void GraphDef::UnsafeArenaSwap(GraphDef* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void GraphDef::InternalSwap(GraphDef* other) {
  using std::swap;
  node_.InternalSwap(&other->node_);
  swap(library_, other->library_);
  swap(versions_, other->versions_);
  swap(version_, other->version_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata GraphDef::GetMetadata() const {
  protobuf_graph_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_graph_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)