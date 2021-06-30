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

#include "operators/unsafe_row_writer_and_reader.h"

namespace sparkcolumnarplugin {
namespace unsaferow {

  int64_t CalculateBitSetWidthInBytes(int32_t numFields) {
    return ((numFields + 63)/ 64) * 8;
  }

  int32_t GetVariableColsNum(std::shared_ptr<arrow::Schema> schema, int64_t num_cols) {
    std::vector<std::shared_ptr<arrow::Field>> fields = schema->fields();
    int32_t count = 0;
    for (auto i = 0; i < num_cols; i ++) {
      auto type_id = fields[i]->type()->id();
      if ((type_id == arrow::BinaryType::type_id) || (type_id == arrow::StringType::type_id)) count++;
    }
    return count;
  }

  arrow::Status UnsafeRowWriterAndReader::Init() {
    printf("call the init method of unsafe row writer and reader");
    int64_t num_rows = rb_->num_rows();
    num_cols_ = rb_->num_columns();
    // Calculate the initial size 
    nullBitsetWidthInBytes_ = CalculateBitSetWidthInBytes(num_cols_);
    int64_t fixedSize = nullBitsetWidthInBytes_ + 8 * num_cols_; // not contain the variable size

    int32_t num_variable_cols = GetVariableColsNum(rb_->schema(), num_cols_);
    
    // Initialize the buffers_ with the initial size.
    for(auto i = 0; i < num_rows; i ++) {
      int64_t new_size = fixedSize + num_variable_cols * 32; // 32 is same with vanilla spark.
      ARROW_ASSIGN_OR_RAISE(
        auto buffer,
        arrow::AllocateResizableBuffer(arrow::BitUtil::BytesForBits(fixedSize),
                                             memory_pool_));
        buffers_[i] = std::move(buffer);
        buffer_cursor_[i] = fixedSize;                         
    }

    row_cursor_ = num_rows - 1;
    
    return arrow::Status::OK();
  }

  bool UnsafeRowWriterAndReader::HasNext() {
    if (row_cursor_ >= 0) {
      return true;
    } else {
      return false;
    }
  }

  std::shared_ptr<arrow::ResizableBuffer> UnsafeRowWriterAndReader::Next(int64_t* length) {
    auto buffer = buffers_[row_cursor_];
    *length = buffer_cursor_[row_cursor_];
    row_cursor_ --;
    return buffer;
  }

  void BitSet(uint8_t* data, int32_t index) {
    int64_t mask = 1L << (index & 0x3f);  // mod 64 and shift
    int64_t wordOffset = (index >> 6) * 8;
    int64_t word[1];
    memcpy(data + wordOffset, word, sizeof(int64_t));
    int64_t value = word[0] | mask;
    memset(data + wordOffset, value, sizeof(int64_t));
  }

  int64_t GetFieldOffset(int64_t nullBitsetWidthInBytes, int32_t index) {
    return nullBitsetWidthInBytes + 8L * index;
  }

  void SetNullAt(uint8_t* data, int64_t offset, int32_t index) {
    BitSet(data, index);
    // set the value to 0
    memset(data + offset, 0, sizeof(int64_t));
    return ;
  }

  int64_t RoundNumberOfBytesToNearestWord(int64_t numBytes) {
    int64_t remainder = numBytes & 0x07;  // This is equivalent to `numBytes % 8`
    if (remainder == 0) {
      return numBytes;
    } else {
      return numBytes + (8 - remainder);
    }
  }

  arrow::Status WriteValue(std::shared_ptr<arrow::ResizableBuffer> buffer, 
    int64_t offset, int32_t index, std::shared_ptr<arrow::Array> array, int64_t currentCursor,
    int64_t* updatedCursor) {
    auto data = buffer->mutable_data();
    // Write the value into the buffer
    switch(array->type_id()) {
      case arrow::BooleanType::type_id:
        {
          // Boolean type
          auto boolArray = std::static_pointer_cast<arrow::BooleanArray>(array);
          memset(data + offset, boolArray->Value(index), sizeof(bool));
          break;
        }
      case arrow::Int8Type::type_id:
        {
          // Byte type
          auto int8Array = std::static_pointer_cast<arrow::Int8Array>(array);
          memset(data + offset, int8Array->Value(index), sizeof(int8_t));
          break;
        }
      case arrow::Int16Type::type_id:
        {
          // Short type
          auto int16Array = std::static_pointer_cast<arrow::Int16Array>(array);
          memset(data + offset, int16Array->Value(index), sizeof(int16_t));
          break;
        }
      case arrow::Int32Type::type_id:
        {
          // Short type
          auto int32Array = std::static_pointer_cast<arrow::Int32Array>(array);
          memset(data + offset, int32Array->Value(index), sizeof(int32_t));
          break;
        }
      case arrow::Int64Type::type_id:
        { 
          // Long type
          auto int64Array = std::static_pointer_cast<arrow::Int64Array>(array);
          memset(data + offset, int64Array->Value(index), sizeof(int64_t));
          break;
        }
      case arrow::FloatType::type_id:
        { 
          // Float type
          auto floatArray = std::static_pointer_cast<arrow::FloatArray>(array);
          memset(data + offset, floatArray->Value(index), sizeof(float));
          break;
        }
      case arrow::DoubleType::type_id:
        { 
          // Double type
          auto doubleArray = std::static_pointer_cast<arrow::DoubleArray>(array);
          memset(data + offset, doubleArray->Value(index), sizeof(double));
          break;
        }
      case arrow::BinaryType::type_id:
        {
          // Binary type
          auto binaryArray = std::static_pointer_cast<arrow::BinaryArray>(array);
          using offset_type = typename arrow::BinaryType::offset_type;
          offset_type length;
          auto value = binaryArray->GetValue(index, &length);

          int64_t roundedSize = RoundNumberOfBytesToNearestWord(length);
          RETURN_NOT_OK(buffer->Resize(roundedSize));
          // TODO: Whether need to zerooutPaddingBytes()
          // write the variable value
          memcpy(data + currentCursor, value, length);
          // write the offset and size
          int64_t offsetAndSize = (currentCursor << 32) | length;
          memset(data + offset, offsetAndSize, sizeof(int64_t));
          // Update the cursor of the buffer.
          *updatedCursor = currentCursor + roundedSize;

          break;
        }
      case arrow::StringType::type_id:
        {
          // String type
          auto stringArray = std::static_pointer_cast<arrow::StringArray>(array);
          using offset_type = typename arrow::StringType::offset_type;
          offset_type length;
          auto value = stringArray->GetValue(index, &length);

          int64_t roundedSize = RoundNumberOfBytesToNearestWord(length);
          RETURN_NOT_OK(buffer->Resize(roundedSize));
          // TODO: Whether need to zerooutPaddingBytes()
          // write the variable value
          memcpy(data + currentCursor, value, length);
          // write the offset and size
          int64_t offsetAndSize = (currentCursor << 32) | length;
          memset(data + offset, offsetAndSize, sizeof(int64_t));
          // Update the cursor of the buffer.
          *updatedCursor = currentCursor + roundedSize;
          break;
        }
      case arrow::Decimal128Type::type_id:
        {
          // decimal128 type
        //   auto binaryArray = std::static_pointer_cast<arrow::Decimal128Array>(array);
        //   memcpy(data + offset, binaryArray->Value(index), length);
        //   break;
        }
      default:
        return arrow::Status::Invalid("Unsupported data type: " + array->type_id());
    }
    return arrow::Status::OK();;
  }

  arrow::Status UnsafeRowWriterAndReader::Write() {
    printf("call the write method of unsafe row writer and reader");
    int64_t num_rows = rb_->num_rows();
    // Get each row value and write to the buffer
    for (auto i = 0; i < num_rows; i++) {
      auto buffer = buffers_[i];
      uint8_t* data = buffer->mutable_data();
      for(auto j = 0; j < num_cols_; j++) {
        auto array = rb_->column(j);
        // for each column, get the current row  value, check whether it is null, and then write it to data.
        bool is_null = array->IsNull(i);
        int64_t offset = GetFieldOffset(nullBitsetWidthInBytes_, j);
        if (is_null) {
          SetNullAt(data, offset, j);
        } else {
          // Write the value to the buffer
          int64_t updatedCursor = buffer_cursor_[i];
          WriteValue(buffer, offset, j, array, buffer_cursor_[i], &updatedCursor);
          buffer_cursor_[i] = updatedCursor;
        }
      }  
    }
    return arrow::Status::OK();
  }

}  // namespace unsaferow
}  // namespace sparkcolumnarplugin
