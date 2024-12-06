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

#include <gtest/gtest.h>
#include "test_utils/DataGen.h"
#include "segcore/SegmentSealed.h"
#include "plan/PlanNode.h"

using namespace milvus;
using namespace milvus::segcore;
using namespace milvus::plan;

class QueryAggTest: public testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        schema_ = std::make_shared<Schema>();
        auto vec_fid = schema_->AddDebugField(
                "fakevec", DataType::VECTOR_FLOAT, 16, knowhere::metric::L2);
        bool nullable = GetParam();
        auto bool_fid = schema_->AddDebugField("bool", DataType::BOOL, nullable);
        auto int8_fid = schema_->AddDebugField("int8", DataType::INT8, nullable);
        auto int16_fid = schema_->AddDebugField("int16", DataType::INT16, nullable);
        auto int32_fid = schema_->AddDebugField("int32", DataType::INT32, nullable);
        auto int64_fid = schema_->AddDebugField("int64", DataType::INT64, nullable);
        auto float_fid = schema_->AddDebugField("float", DataType::FLOAT, nullable);
        auto double_fid = schema_->AddDebugField("double", DataType::DOUBLE, nullable);
        auto str1_fid = schema_->AddDebugField("string1", DataType::VARCHAR, nullable);
        schema_->set_primary_field_id(str1_fid);

        auto segment = CreateSealedSegment(schema_);
        num_rows_ = 100;
        auto raw_data = DataGen(schema_, num_rows_, 42, 0, 1, 10, false, true, nullable);
        auto fields = schema_->get_fields();
        for (auto field_data : raw_data.raw_->fields_data()) {
            int64_t field_id = field_data.field_id();

            auto info = FieldDataInfo(field_data.field_id(), num_rows_, "/tmp/a");
            auto field_meta = fields.at(FieldId(field_id));
            info.channel->push(
                    CreateFieldDataFromDataArray(num_rows_, &field_data, field_meta));
            info.channel->close();

            segment->LoadFieldData(FieldId(field_id), info);
        }
        segment_ = SegmentSealedSPtr(segment.release());
    }

    void TearDown() override {

    }
public:
    int64_t num_rows_{0};
    SegmentSealedSPtr segment_;
    std::shared_ptr<Schema> schema_;
};

INSTANTIATE_TEST_SUITE_P(TaskTestSuite,
                         QueryAggTest,
                         ::testing::Values(true,
                                           false));


TEST_P(QueryAggTest, GroupFixedLengthType) {
    std::cout << "hc=== query agg test" << std::endl;
    std::vector<milvus::plan::PlanNodePtr> sources;
    //set up mvcc_node + project_node + agg_node
    PlanNodePtr mvcc_node = std::make_shared<milvus::plan::MvccNode>(
            milvus::plan::GetNextPlanNodeId(), sources);


}