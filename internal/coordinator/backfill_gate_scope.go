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

package coordinator

// NewWatermarkScope builds the scope of an internal (add_function_field) round: every
// sealed segment with schema_version < schemaVersion still needs the backfill, growing
// segments born before schemaChangeTimeTick may hold pre-V rows, and pre-tick sealed
// segments need a Finished index on each of vectorFields before the round may serve.
func NewWatermarkScope(schemaVersion int32, schemaChangeTimeTick uint64, vectorFields []int64) BackfillScope {
	return BackfillScope{
		Kind:                 ScopeWatermark,
		Watermark:            schemaVersion,
		SchemaChangeTimeTick: schemaChangeTimeTick,
		VectorFields:         vectorFields,
	}
}
