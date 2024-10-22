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

#include "common/Vector.h"
#include "common/Types.h"

namespace milvus {
namespace exec {


class VectorHasher{
public:
    VectorHasher(DataType data_type, column_index_t column_idx)
        :channel_type_(data_type), channel_idx_(column_idx){}

    column_index_t ChannelIndex() const {
        return channel_idx_;
    }

    DataType ChannelDataType() const {
        return channel_type_;
    }

private:
    const column_index_t channel_idx_;
    const DataType channel_type_;
};
}
}

