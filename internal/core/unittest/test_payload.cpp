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
#include "storage/PayloadWriter.h"
#include "storage/PayloadReader.h"
#include "storage/PayloadStream.h"
#include <random>

TEST(Payload, TestRead) {
    const int length = 100;
    uint8_t data[length];
    for (auto i = 0; i < length; i++) {
        data[i] = i%256;
    }
    milvus::DataType data_type = milvus::DataType::BINARY;
    auto nullable = false;
    auto is_field_data = false;
    auto payload_reader = std::make_shared<milvus::storage::PayloadReader>(data, length, data_type, nullable, is_field_data);
    std::cout << "init payload_reader success" << std::endl;
    ASSERT_EQ(payload_reader->get_length(), length);
}

TEST(Payload, BinaryVsInt8) {
    // data
    uint64_t seed = 42;
    std::default_random_engine random(seed);
    const int length = 1000000;
    uint8_t data[length];
    bool random_val = false;
    for (auto i = 0; i < length; i++) {
        if (random_val)
            data[i] = random()%256;
        else
            data[i] = i%256;
    }
    //Test PayloadWriter binary
    {
        milvus::DataType data_type = milvus::DataType::BINARY;
        auto payload_writer = std::make_unique<milvus::storage::PayloadWriter>(data_type, false);
        payload_writer->add_one_binary_payload(data, length);
        payload_writer->finish();
        auto payload_buffer = payload_writer->get_payload_buffer();
        std::cout << "hc==binary payload size:" << payload_buffer.size() << std::endl;
    }
    //Test payloadWriter int8
    {
        milvus::DataType data_type = milvus::DataType::INT8;
        auto payload_writer = std::make_unique<milvus::storage::PayloadWriter>(data_type, false);
        auto payload =
                milvus::storage::Payload{data_type,
                                         data,
                                         nullptr,
                                         length,
                                         1,
                                         false};
        payload_writer->add_payload(payload);
        payload_writer->finish();
        auto payload_buffer = payload_writer->get_payload_buffer();
        std::cout << "hc==int8 payload size:" << payload_buffer.size() << std::endl;
    }
}
