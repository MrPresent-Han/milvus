// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <memory>
#include <vector>

#include "storage/DataCodec.h"
#include "storage/Types.h"

namespace milvus::storage {

// TODO :: indexParams storage in a single file
class IndexData : public DataCodec {
 public:
    explicit IndexData(FieldDataPtr data)
        : DataCodec(data, CodecType::IndexDataType) {
    }

    explicit IndexData(Slice& index_slice): DataCodec(CodecType::IndexDataType)  {
        index_slice_ = index_slice;
    }

    std::vector<uint8_t>
    Serialize(StorageType medium) override;

    void
    SetFieldDataMeta(const FieldDataMeta& meta) override;

 public:
    void
    set_index_meta(const IndexMeta& meta);

    std::vector<uint8_t>
    serialize_to_remote_file();

    std::vector<uint8_t>
    serialize_to_local_file();

    const uint8_t*
    IndexBin() {
        AssertInfo(index_slice_.has_value(), "failed to get index slice from null value index_bin_");
        auto index_slice = index_slice_.value();
        return index_slice.Data();
    }

    int64_t
    IndexBinSize(){
        AssertInfo(index_slice_.has_value(), "failed to get index bin from null value index_bin_");
        return index_slice_.value().Size();
    }

 private:
    std::optional<FieldDataMeta> field_data_meta_;
    std::optional<IndexMeta> index_meta_;
    std::optional<Slice> index_slice_;
};

}  // namespace milvus::storage
