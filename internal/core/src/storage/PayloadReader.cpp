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

#include "arrow/io/api.h"
#include "arrow/status.h"
#include "common/EasyAssert.h"
#include "common/Types.h"
#include "parquet/arrow/reader.h"
#include "parquet/column_reader.h"
#include "storage/PayloadReader.h"
#include "storage/Util.h"

namespace milvus::storage {

PayloadReader::PayloadReader(const uint8_t* data,
                             int length,
                             DataType data_type,
                             bool nullable,
                             bool is_field_data)
    : column_type_(data_type), nullable_(nullable), length_(length){
    LOG_INFO("hc===init PayloadReader, length:{}", length);
    auto input = std::make_shared<arrow::io::BufferReader>(data, length);
    init(input, is_field_data);
}

void
PayloadReader::init(std::shared_ptr<arrow::io::BufferReader> input,
                    bool is_field_data) {
    LOG_INFO("hc===start to init payloadReader");
    arrow::MemoryPool* pool = arrow::default_memory_pool();

    // Configure general Parquet reader settings
    auto reader_properties = parquet::ReaderProperties(pool);
    reader_properties.set_buffer_size(4096 * 4);
    // reader_properties.enable_buffered_stream();

    // Configure Arrow-specific Parquet reader settings
    auto arrow_reader_props = parquet::ArrowReaderProperties();
    arrow_reader_props.set_batch_size(128 * 1024);  // default 64 * 1024
    arrow_reader_props.set_pre_buffer(false);

    parquet::arrow::FileReaderBuilder reader_builder;
    auto st = reader_builder.Open(input, reader_properties);
    AssertInfo(st.ok(), "file to read file");
    reader_builder.memory_pool(pool);
    reader_builder.properties(arrow_reader_props);

    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    st = reader_builder.Build(&arrow_reader);
    AssertInfo(st.ok(), "build file reader");
    LOG_INFO("hc===set up arrow read");
    int64_t column_index = 0;
    auto file_meta = arrow_reader->parquet_reader()->metadata();

    // dim is unused for sparse float vector
    dim_ = (IsVectorDataType(column_type_) &&
            !IsSparseFloatVectorDataType(column_type_))
               ? GetDimensionFromFileMetaData(
                     file_meta->schema()->Column(column_index), column_type_)
               : 1;
    auto total_num_rows = file_meta->num_rows();

    std::shared_ptr<::arrow::RecordBatchReader> rb_reader;
    st = arrow_reader->GetRecordBatchReader(&rb_reader);
    AssertInfo(st.ok(), "get record batch reader");
    if (is_field_data) {
        if (column_type_ == milvus::DataType::BINARY) {
            LOG_INFO("hc===init payloadreader in binary type");
            std::shared_ptr<arrow::RecordBatch> record_batch;
            rb_reader->ReadNext(&record_batch);
            std::shared_ptr<arrow::Array> array = record_batch->column(column_index);
            AssertInfo(array->type_id()==arrow::Type::BINARY, "inconsistent array type for reading slice");
            auto binary_array = std::dynamic_pointer_cast<arrow::BinaryArray>(array);

            // get the first array
            auto array_length = 0;
            const uint8_t* first_binary_slice = binary_array->GetValue(0, &array_length);
            AssertInfo(first_binary_slice!=nullptr && array_length>0, "Invalid binary from payload");

            std::shared_ptr<uint8_t[]> copied_data = std::shared_ptr<uint8_t[]>(new uint8_t[array_length]);
            std::memcpy(copied_data.get(), first_binary_slice, array_length);
            slice_ = Slice(copied_data, array_length);
            LOG_INFO("hc===set up init payloadreader in binary type, slice_pointer:{}, slice_size:{}, array_length:{}",
                     slice_.value().Data()!=nullptr, slice_.value().Size(), array_length);
        } else {
            LOG_INFO("hc===init payloadreader in non-binary type with fieldData");
            field_data_ =
                    CreateFieldData(column_type_, nullable_, dim_, total_num_rows);
            for (arrow::Result<std::shared_ptr<arrow::RecordBatch>> maybe_batch :
                    *rb_reader) {
                AssertInfo(maybe_batch.ok(), "get batch record success");
                auto array = maybe_batch.ValueOrDie()->column(column_index);
                // to read
                field_data_->FillFieldData(array);
            }
            AssertInfo(field_data_->IsFull(), "field data hasn't been filled done");
            LOG_INFO("hc===finish init payloadreader in non-binary type with fieldData");
        }
    } else {
        arrow_reader_ = std::move(arrow_reader);
        record_batch_reader_ = std::move(rb_reader);
    }
}

}  // namespace milvus::storage
