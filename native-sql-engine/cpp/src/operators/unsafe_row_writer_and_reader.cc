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

#include <iostream>

namespace sparkcolumnarplugin {
namespace unsaferow {

  int64_t CalculateBitSetWidthInBytes(int32_t numFields) {
    return ((numFields + 63) / 64) * 8;
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
    num_rows_ = rb_->num_rows();
    num_cols_ = rb_->num_columns();
 
    // Calculate the initial size 
    nullBitsetWidthInBytes_ = CalculateBitSetWidthInBytes(num_cols_);
    
    int64_t fixedSize = nullBitsetWidthInBytes_ + 8 * num_cols_; // not contain the variable size

    int32_t num_variable_cols = GetVariableColsNum(rb_->schema(), num_cols_);
   
    // Initialize the buffers_ with the initial size.
    for(auto i = 0; i < num_rows_; i ++) {
      int64_t new_size = fixedSize + num_variable_cols * 32; // 32 is same with vanilla spark.
      //  std::cout << "the fixedSize is " << fixedSize << "the new size is " << new_size << "\n" << std::flush;
      std::shared_ptr<arrow::ResizableBuffer> buffer;
      ARROW_ASSIGN_OR_RAISE(
        buffer,
        arrow::AllocateResizableBuffer(arrow::BitUtil::BytesForBits(new_size * 8),
                                             memory_pool_));                                                               
      memset(buffer->mutable_data(), 0, new_size);                                     
      buffer_cursor_.push_back(fixedSize);                                        
      buffers_.push_back(std::move(buffer));                                                  
    }

    row_cursor_ = 0;
    return arrow::Status::OK();
  }

  bool UnsafeRowWriterAndReader::HasNext() {
    if (row_cursor_ < num_rows_) {
      return true;
    } else {
      return false;
    }
  }

  arrow::Status  UnsafeRowWriterAndReader::Next(int64_t* length, std::shared_ptr<arrow::ResizableBuffer>* buffer) {
    *buffer = buffers_[row_cursor_];
    *length = buffer_cursor_[row_cursor_];
    row_cursor_ ++;
    return arrow::Status::OK();
  }

  void BitSet(uint8_t* data, int32_t index) {
    int64_t mask = 1L << (index & 0x3f);  // mod 64 and shift
    int64_t wordOffset = (index >> 6) * 8;
    int64_t word;
    memcpy(&word, data + wordOffset, sizeof(int64_t));
    int64_t value = word | mask;
    memcpy(data + wordOffset, &value, sizeof(int64_t));
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
    int64_t offset, int32_t row_index, std::shared_ptr<arrow::Array> array, int64_t currentCursor,
    int64_t* updatedCursor) {
    auto data = buffer->mutable_data();
    // Write the value into the buffer
    switch(array->type_id()) {
      case arrow::BooleanType::type_id:
        {
          // Boolean type
          auto boolArray = std::static_pointer_cast<arrow::BooleanArray>(array);
          std::cout << "begin set the boolean type" << "\n";
          auto value = boolArray->Value(row_index);
          memcpy(data + offset, &value, sizeof(bool));
          std::cout << "end set the boolean type" << "\n";
          break;
        }
      case arrow::Int8Type::type_id:
        {
          // Byte type
          auto int8Array = std::static_pointer_cast<arrow::Int8Array>(array);
          std::cout << "begin set the byte type" << "\n";
          auto value = int8Array->Value(row_index);
          memcpy(data + offset, &value, sizeof(int8_t));
          std::cout << "end set the byte type" << "\n";
          break;
        }
      case arrow::Int16Type::type_id:
        {
          // Short type
          auto int16Array = std::static_pointer_cast<arrow::Int16Array>(array);
          std::cout << "begin set the short type" << "\n";
          auto value = int16Array->Value(row_index);
          memcpy(data + offset, &value, sizeof(int16_t));
          std::cout << "end set the short type" << "\n";
          break;
        }
      case arrow::Int32Type::type_id:
        {
          // Integer type
          auto int32Array = std::static_pointer_cast<arrow::Int32Array>(array);
          std::cout << "begin set the Integer type" << "\n";
          auto value = int32Array->Value(row_index);
          memcpy(data + offset, &value, sizeof(int32_t));
          std::cout << "end set the Integer type" << "\n";
          break;
        }
      case arrow::Int64Type::type_id:
        { 
          // Long type
          auto int64Array = std::static_pointer_cast<arrow::Int64Array>(array);
          std::cout << "begin set the Long type" << "\n";
          auto value = int64Array->Value(row_index);
          memcpy(data + offset, &value, sizeof(int64_t));
          std::cout << "end set the Long type" << "\n";
          break;
        }
      case arrow::FloatType::type_id:
        { 
          // Float type
          auto floatArray = std::static_pointer_cast<arrow::FloatArray>(array);
          std::cout << "begin set the Float type" << "\n";
          auto value = floatArray->Value(row_index);
          memcpy(data + offset, &value, sizeof(float));
          std::cout << "end set the Float type" << "\n";
          break;
        }
      case arrow::DoubleType::type_id:
        { 
          // Double type
          auto doubleArray = std::static_pointer_cast<arrow::DoubleArray>(array);
          std::cout << "begin set the Double type" << "\n";
          auto value = doubleArray->Value(row_index);
          memcpy(data + offset, &value, sizeof(double));
          std::cout << "end set the Double type" << "\n";
          break;
        }
      case arrow::BinaryType::type_id:
        {
          // Binary type
          std::cout << "begin set the binary type" << "\n";
          auto binaryArray = std::static_pointer_cast<arrow::BinaryArray>(array);
          using offset_type = typename arrow::BinaryType::offset_type;
          offset_type length;
          auto value = binaryArray->GetValue(row_index, &length);
          int64_t roundedSize = RoundNumberOfBytesToNearestWord(length);
          if (roundedSize > 32) {
            // after call the resize method , the buffer is wrong.
            RETURN_NOT_OK(buffer->Resize(roundedSize));
          }
          // After resize buffer, the data address is changed and the value is reset to 0.
          auto new_data = buffer->mutable_data();
          // write the variable value
          memcpy(new_data + currentCursor, value, length);
          // write the offset and size
          int64_t offsetAndSize = (currentCursor << 32) | length;
          memcpy(new_data + offset, &offsetAndSize, sizeof(int64_t));
          // Update the cursor of the buffer.
          *updatedCursor = currentCursor + roundedSize;
           std::cout << "end set the binary type" << "\n";
          break;
        }
      case arrow::StringType::type_id:
        {
          // String type
          std::cout << "begin set the string type" << "\n";
          auto stringArray = std::static_pointer_cast<arrow::StringArray>(array);
          using offset_type = typename arrow::StringType::offset_type;
          offset_type length;
          auto value = stringArray->GetValue(row_index, &length);
          int64_t roundedSize = RoundNumberOfBytesToNearestWord(length);
        
          if (roundedSize > 32) {
            // after call the resize method , the buffer is wrong.
            RETURN_NOT_OK(buffer->Resize(roundedSize));
          }
      
          // After resize buffer, the data address is changed and the value is reset to 0.
          auto new_data = buffer->mutable_data();
          // write the variable value
          memcpy(new_data + currentCursor, value, length);
          // write the offset and size
          int64_t offsetAndSize = (currentCursor << 32) | length;
          memcpy(new_data + offset, &offsetAndSize, sizeof(int64_t));
          // Update the cursor of the buffer.
          *updatedCursor = currentCursor + roundedSize;
          std::cout << "end set the string type" << "\n";
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
    int64_t num_rows = rb_->num_rows();
    // Get each row value and write to the buffer
    for (auto i = 0; i < num_rows; i++) {
      auto buffer = buffers_[i];
      for(auto j = 0; j < num_cols_; j++) {
        uint8_t* data = buffer->mutable_data();
        auto array = rb_->column(j);
        // for each column, get the current row  value, check whether it is null, and then write it to data.
        bool is_null = array->IsNull(i);
        int64_t offset = GetFieldOffset(nullBitsetWidthInBytes_, j);
        if (is_null) {
          SetNullAt(data, offset, j);
        } else {
          // Write the value to the buffer
          int64_t updatedCursor = buffer_cursor_[i];
          WriteValue(buffer, offset, i, array, buffer_cursor_[i], &updatedCursor);
          buffer_cursor_[i] = updatedCursor;
        }
      }  
    }
    return arrow::Status::OK();
  }

}  // namespace unsaferow
}  // namespace sparkcolumnarplugin
