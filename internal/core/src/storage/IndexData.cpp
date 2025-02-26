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

#include "storage/IndexData.h"
#include "common/EasyAssert.h"
#include "common/Consts.h"
#include "storage/Event.h"
#include "log/Log.h"

namespace milvus::storage {

void
IndexData::SetFieldDataMeta(const FieldDataMeta& meta) {
    AssertInfo(!field_data_meta_.has_value(), "field meta has been inited");
    field_data_meta_ = meta;
}

void
IndexData::set_index_meta(const IndexMeta& meta) {
    AssertInfo(!index_meta_.has_value(), "index meta has been inited");
    index_meta_ = meta;
}

std::vector<uint8_t>
IndexData::Serialize(StorageType medium) {
    switch (medium) {
        case StorageType::Remote:
            return serialize_to_remote_file();
        case StorageType::LocalDisk:
            return serialize_to_local_file();
        default:
            PanicInfo(DataFormatBroken,
                      fmt::format("unsupported medium type {}", medium));
    }
}

std::vector<uint8_t>
IndexData::serialize_to_remote_file() {
    AssertInfo(index_slice_.has_value(), "index slice should have value for index data");
    //1. set up descriptor_event
    DescriptorEvent descriptor_event;
    auto& des_event_data = descriptor_event.event_data;
    auto& des_fix_part = des_event_data.fix_part;
    des_fix_part.collection_id = field_data_meta_->collection_id;
    des_fix_part.partition_id = field_data_meta_->partition_id;
    des_fix_part.segment_id = field_data_meta_->segment_id;
    des_fix_part.field_id = field_data_meta_->field_id;
    des_fix_part.start_timestamp = time_range_.first;
    des_fix_part.end_timestamp = time_range_.second;
    des_fix_part.data_type = milvus::proto::schema::DataType::Binary;
    for (auto i = int8_t(EventType::DescriptorEvent);
         i < int8_t(EventType::EventTypeEnd);
         i++) {
        des_event_data.post_header_lengths.push_back(
                GetEventFixPartSize(EventType(i)));
    }
    des_event_data.extras[ORIGIN_SIZE_KEY] =
            std::to_string(index_slice_.value().Size());
    des_event_data.extras[INDEX_BUILD_ID_KEY] =
            std::to_string(index_meta_->build_id);
    auto& des_event_header = descriptor_event.event_header;
    des_event_header.timestamp_ = 0;
    auto des_event_bytes = descriptor_event.Serialize();
    auto des_size = des_event_bytes.size();

    //2. set up index event
    IndexEvent index_event;
    index_event.event_offset = des_event_bytes.size();
    auto& index_event_data = index_event.event_data;
    index_event_data.start_timestamp = time_range_.first;
    index_event_data.end_timestamp = time_range_.second;
    index_event_data.slice_ = index_slice_.value();

    auto& index_event_header = index_event.event_header;
    index_event_header.event_type_ = EventType::IndexFileEvent;
    index_event_header.timestamp_ = 0;
    auto index_event_bytes = index_event.Serialize();
    des_event_bytes.insert(des_event_bytes.end(),
                           index_event_bytes.begin(),
                           index_event_bytes.end());
    LOG_INFO("hc, index_data_des_size:{}, index_data_size:{}", des_size, index_event_bytes.size());
    return des_event_bytes;
}

// Just for test
std::vector<uint8_t>
IndexData::serialize_to_local_file() {
    LocalIndexEvent event;
    event.field_data = field_data_;

    return event.Serialize();
}

}  // namespace milvus::storage
