// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include "common/BitsetView.h"
#include "query/PlanNode.h"
#include "query/SearchOnGrowing.h"
#include "segcore/SealedIndexingRecord.h"

namespace milvus::segcore{
    class SegmentInternalInterface;
}

namespace milvus::query {

struct SearchContext{
    const SearchInfo& search_info_;
    const BitsetView& bitset_;
    const segcore::SegmentInternalInterface& segment_;

    SearchContext(const SearchInfo& search_info,
                  const BitsetView& bitset,
                  const segcore::SegmentInternalInterface& segment):
                  search_info_(search_info),
                  bitset_(bitset),
                  segment_(segment){}
};

void
SearchOnSealedIndex(const Schema& schema,
                    const segcore::SealedIndexingRecord& record,
                    const void* query_data,
                    int64_t num_queries,
                    SearchResult& result,
                    const SearchContext& searchContext);

void
SearchOnSealed(const Schema& schema,
               const void* vec_data,
               const SearchInfo& search_info,
               const void* query_data,
               int64_t num_queries,
               int64_t row_count,
               const BitsetView& bitset,
               SearchResult& result);

}  // namespace milvus::query
