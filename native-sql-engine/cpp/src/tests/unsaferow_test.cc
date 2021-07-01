/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <arrow/compute/api.h>
#include <arrow/datum.h>
#include <arrow/io/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/util.h>
#include <arrow/record_batch.h>
#include <arrow/util/io_util.h>
#include <gtest/gtest.h>

#include <iostream>

#include "operators/unsafe_row_writer_and_reader.h"
#include "tests/test_utils.h"

namespace sparkcolumnarplugin {
namespace unsaferow {

class MyMemoryPool : public arrow::MemoryPool {
 public:
  explicit MyMemoryPool(int64_t capacity) : capacity_(capacity) {}

  Status Allocate(int64_t size, uint8_t** out) override {
    if (bytes_allocated() + size > capacity_) {
      return Status::OutOfMemory("malloc of size ", size, " failed");
    }
    RETURN_NOT_OK(pool_->Allocate(size, out));
    stats_.UpdateAllocatedBytes(size);
    return arrow::Status::OK();
  }

  Status Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) override {
    if (new_size > capacity_) {
      return Status::OutOfMemory("malloc of size ", new_size, " failed");
    }
    RETURN_NOT_OK(pool_->Reallocate(old_size, new_size, ptr));
    stats_.UpdateAllocatedBytes(new_size - old_size);
    return arrow::Status::OK();
  }

  void Free(uint8_t* buffer, int64_t size) override {
    pool_->Free(buffer, size);
    stats_.UpdateAllocatedBytes(-size);
  }

  int64_t bytes_allocated() const override { return stats_.bytes_allocated(); }

  int64_t max_memory() const override { return pool_->max_memory(); }

  std::string backend_name() const override { return pool_->backend_name(); }

 private:
  MemoryPool* pool_ = arrow::default_memory_pool();
  int64_t capacity_;
  arrow::internal::MemoryPoolStats stats_;
};

class SplitterTest : public ::testing::Test {
 protected:
  void SetUp() {
    auto f_na = field("f_na", arrow::null());
    auto f_int8_a = field("f_int8_a", arrow::int8());
    auto f_int8_b = field("f_int8_b", arrow::int8());
    auto f_int32 = field("f_int32", arrow::int32());
    auto f_uint64 = field("f_uint64", arrow::int64());
    auto f_double = field("f_double", arrow::float64());
    auto f_bool = field("f_bool", arrow::boolean());
    auto f_string = field("f_string", arrow::utf8());
    auto f_nullable_string = field("f_nullable_string", arrow::utf8());
    auto f_decimal = field("f_decimal128", arrow::decimal(10, 2));

    ARROW_ASSIGN_OR_THROW(tmp_dir_1_,
                          std::move(arrow::internal::TemporaryDir::Make(tmp_dir_prefix)))
    ARROW_ASSIGN_OR_THROW(tmp_dir_2_,
                          std::move(arrow::internal::TemporaryDir::Make(tmp_dir_prefix)))
    auto config_dirs =
        tmp_dir_1_->path().ToString() + "," + tmp_dir_2_->path().ToString();

    setenv("NATIVESQL_SPARK_LOCAL_DIRS", config_dirs.c_str(), 1);

    // schema_ = arrow::schema({f_na, f_int8_a, f_int8_b, f_int32, f_uint64, f_double,
    //                          f_bool, f_string, f_nullable_string, f_decimal});

    schema_ = arrow::schema({f_uint64});

    // MakeInputBatch(input_data_1, schema_, &input_batch_1_);
    MakeInputBatch(input_data_2, schema_, &input_batch_2_);
  }

  void TearDown() override {
    if (file_ != nullptr && !file_->closed()) {
      file_->Close();
    }
  }

  static void CheckFileExsists(const std::string& file_name) {
    ASSERT_EQ(*arrow::internal::FileExists(
                  *arrow::internal::PlatformFilename::FromString(file_name)),
              true);
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> TakeRows(
      const std::shared_ptr<arrow::RecordBatch>& input_batch,
      const std::string& json_idx) {
    std::shared_ptr<arrow::Array> take_idx;
    ASSERT_NOT_OK(
        arrow::ipc::internal::json::ArrayFromJSON(arrow::int32(), json_idx, &take_idx));

    auto cntx = arrow::compute::ExecContext();
    std::shared_ptr<arrow::RecordBatch> res;
    ARROW_ASSIGN_OR_RAISE(
        arrow::Datum result,
        arrow::compute::Take(arrow::Datum(input_batch), arrow::Datum(take_idx),
                             arrow::compute::TakeOptions{}, &cntx));
    return result.record_batch();
  }

  arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchReader>>
  GetRecordBatchStreamReader(const std::string& file_name) {
    if (file_ != nullptr && !file_->closed()) {
      RETURN_NOT_OK(file_->Close());
    }
    ARROW_ASSIGN_OR_RAISE(file_, arrow::io::ReadableFile::Open(file_name))
    ARROW_ASSIGN_OR_RAISE(auto file_reader,
                          arrow::ipc::RecordBatchStreamReader::Open(file_))
    return file_reader;
  }

  static const std::string tmp_dir_prefix;
  static const std::vector<std::string> input_data_1;
  static const std::vector<std::string> input_data_2;

  std::shared_ptr<arrow::internal::TemporaryDir> tmp_dir_1_;
  std::shared_ptr<arrow::internal::TemporaryDir> tmp_dir_2_;

  std::shared_ptr<arrow::Schema> schema_;

  std::shared_ptr<arrow::RecordBatch> input_batch_1_;
  std::shared_ptr<arrow::RecordBatch> input_batch_2_;

  std::shared_ptr<arrow::io::ReadableFile> file_;
};

const std::string SplitterTest::tmp_dir_prefix = "columnar-shuffle-test";
const std::vector<std::string> SplitterTest::input_data_1 = {
    "[null, null, null, null, null, null, null, null, null, null]",
    "[1, 2, 3, null, 4, null, 5, 6, null, 7]",
    "[1, -1, null, null, -2, 2, null, null, 3, -3]",
    "[1, 2, 3, 4, null, 5, 6, 7, 8, null]",
    "[null, null, null, null, null, null, null, null, null, null]",
    R"([-0.1234567, null, 0.1234567, null, -0.142857, null, 0.142857, 0.285714, 0.428617, null])",
    "[null, true, false, null, true, true, false, true, null, null]",
    R"(["alice0", "bob1", "alice2", "bob3", "Alice4", "Bob5", "AlicE6", "boB7", "ALICE8", "BOB9"])",
    R"(["alice", "bob", null, null, "Alice", "Bob", null, "alicE", null, "boB"])",
    R"(["-1.01", "2.01", "-3.01", null, "0.11", "3.14", "2.27", null, "-3.14", null])"};

const std::vector<std::string> SplitterTest::input_data_2 = {
    "[1]"};

TEST_F(SplitterTest, TestUnsaferowWriterandReader) { 
     std::shared_ptr<arrow::MemoryPool> pool = std::make_shared<MyMemoryPool>(4000);
    std::shared_ptr<UnsafeRowWriterAndReader> unsafe_row_writer_reader = 
      std::make_shared<UnsafeRowWriterAndReader>(input_batch_2_, pool.get());

      std::shared_ptr<Schema> schema = input_batch_2_->schema();
      unsafe_row_writer_reader->Init();
      unsafe_row_writer_reader->Write();
}
}  // namespace shuffle
}  // namespace sparkcolumnarplugin
